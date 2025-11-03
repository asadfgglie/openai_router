from typing import Annotated
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
import httpx
from loguru import logger
from contextlib import asynccontextmanager
import gradio as gr
import pandas as pd
from datetime import datetime, timezone
import typer
import uvicorn
import webbrowser
import time

# --- 导入 SQLModel 和同步组件 ---
# 切换到同步 Session 和 Engine
from sqlmodel import Field, SQLModel, create_engine, Session, select
from sqlalchemy.engine import Engine  # 导入同步 Engine

# --- 数据库配置 ---
SQLITE_DB_FILE = "routes.db"
# 使用同步 SQLite URL
SQLITE_URL = f"sqlite:///{SQLITE_DB_FILE}"

# --- 全局变量 ---
client: httpx.AsyncClient = None
# 同步 Engine
engine: Engine = None


# --- SQLModel 数据模型 ---
class ModelRoute(SQLModel, table=True):
    """
    一个 SQLModel 模型，用于存储模型名称到后端 URL 的路由。
    'model_name' 是主键，确保了唯一性。
    """

    model_name: str = Field(primary_key=True, index=True)
    model_url: str
    api_key: str | None = Field(default=None)
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )


# --- 同步数据库辅助函数 (将在线程池中执行) ---


def create_db_and_tables_sync():
    """同步创建数据库和表"""
    SQLModel.metadata.create_all(engine)


def get_all_model_names_sync():
    """同步地从数据库中查询所有可用的模型名称"""
    with Session(engine) as session:
        statement = select(ModelRoute.model_name)
        results = session.exec(statement)
        available_models = results.all()
    return available_models


def get_all_routes_sync():
    """同步地从数据库中查询所有可用的 ModelRoute 记录"""
    with Session(engine) as session:
        # 查询 ModelRoute 的所有记录
        statement = select(ModelRoute)
        results = session.exec(statement)
        # 返回 ModelRoute 对象的列表
        all_routes = results.all()
    return all_routes


def get_routing_info_sync(model: str):
    """同步地从数据库中查询模型和所有可用模型"""
    with Session(engine) as session:
        # 1. 获取特定模型
        db_route = session.get(ModelRoute, model)

        # 2. 获取所有可用模型
        available_models = get_all_model_names_sync()
        server = db_route.model_url if db_route else None
        # 3. 获取后端 API 密钥
        backend_api_key = db_route.api_key if db_route else None

        return server, available_models, backend_api_key


# --- FastAPI 生命周期 ---

import os
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError, DBAPIError


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, engine

    # --- 启动时的逻辑 ---

    # 1. 初始化数据库引擎 (同步)
    temp_engine = create_engine(SQLITE_URL, echo=False)
    # ---------- 检查数据库 schema 是否与 SQLModel 定义匹配 ----------
    try:
        # 1. 获取 Inspector
        inspector = inspect(temp_engine)

        # 2. 获取数据库中 'modelroute' 表的实际列
        # 如果表不存在，这将触发 OperationalError，并被 catch 块捕获
        actual_columns_info = inspector.get_columns(ModelRoute.__tablename__)
        actual_columns = {col["name"] for col in actual_columns_info}

        # 3. 获取 SQLModel 中定义的预期列
        expected_columns = {col.name for col in ModelRoute.__table__.columns}

        # 4. 比较
        if actual_columns != expected_columns:
            logger.warning(
                f"数据库 Schema 不匹配! 预期: {expected_columns}, 实际: {actual_columns}"
            )
            logger.warning(f"正在删除旧数据库文件: {SQLITE_DB_FILE} 以进行重建...")

            # 必须先 dispose，释放所有连接，然后才能删除文件
            temp_engine.dispose()

            try:
                os.remove(SQLITE_DB_FILE)
                logger.info(f"成功删除旧数据库: {SQLITE_DB_FILE}")
            except OSError as e:
                logger.error(
                    f"无法删除数据库文件 {SQLITE_DB_FILE}: {e}. 请手动删除它。"
                )
                # 即使删除失败，我们也尝试重新创建引擎

            # 重新创建 engine 实例，以便连接到新文件
            temp_engine = create_engine(SQLITE_URL, echo=False)

    except (OperationalError, DBAPIError) as e:
        # 这是预期的“错误”，如果数据库文件或 'modelroute' 表还不存在
        # 我们安全地忽略它，然后继续创建表
        logger.info(f"数据库或表 '{ModelRoute.__tablename__}' 未找到。将创建新表。")
    except Exception as e:
        # 捕获任何其他意外的检查错误
        logger.error(f"数据库 schema 检查期间发生意外错误: {e}")
        logger.warning("将尝试继续，但可能会失败。")
    engine = temp_engine
    # ---------- 检查数据库 schema 是否与 SQLModel 定义匹配 ----------

    # 2. 创建数据库和表 (使用 run_in_threadpool 执行同步代码)
    await run_in_threadpool(create_db_and_tables_sync)

    # 3. 初始化 httpx 客户端
    timeout = httpx.Timeout(10.0, connect=60.0, read=None, write=60.0)
    client = httpx.AsyncClient(timeout=timeout)

    yield  # 应用运行期间

    # --- 关闭时的逻辑 ---

    if client:
        await client.aclose()
        logger.info("HTTPX client closed.")

    if engine:
        engine.dispose()
        logger.info("Database engine disposed.")


# --- FastAPI 核心应用 ---

# 使用 lifespan 创建 FastAPI 实例
app = FastAPI(lifespan=lifespan)


async def _get_routing_info(request: Request):
    """
    异步辅助函数：解析请求体，并在线程池中执行同步数据库查询。
    """
    try:
        json_body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = json_body.get("model")
    if model is None:
        raise HTTPException(
            status_code=400, detail="'model' field is required in request body"
        )

    # 在线程池中执行同步查询
    server, available_models, backend_api_key = await run_in_threadpool(
        get_routing_info_sync, model
    )

    if server is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available models: {available_models}",
        )

    backend_url = server + request.url.path
    logger.info(f"Routing to backend_url: {backend_url} for model {model}")

    return backend_url, json_body, backend_api_key


async def _stream_proxy(
    backend_url: str, request: Request, json_body: dict, backend_api_key: str | None
):
    """
    这是一个异步生成器，用于代理流式响应。
    """
    headers = {
        h: v
        for h, v in request.headers.items()
        if h.lower() not in ["host", "content-length"]
    }
    if backend_api_key:
        headers["Authorization"] = f"Bearer {backend_api_key}"
    try:
        async with client.stream(
            request.method,
            backend_url,
            params=request.query_params,
            json=json_body,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                error_content = await response.aread()
                logger.warning(
                    f"Backend error: {response.status_code} - {error_content.decode()}"
                )
                raise HTTPException(
                    status_code=response.status_code, detail=error_content.decode()
                )

            async for chunk in response.aiter_bytes():
                yield chunk

    except httpx.ConnectError as e:
        logger.error(f"Connection error to backend {backend_url}: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")
    except Exception as e:
        logger.error(f"An error occurred during streaming proxy: {e}")
        logger.error("Streaming interrupted due to an error.")


async def _non_stream_proxy(
    backend_url: str, request: Request, json_body: dict, backend_api_key: str | None
):
    """
    处理非流式请求的代理逻辑。
    """
    headers = {
        h: v
        for h, v in request.headers.items()
        if h.lower() not in ["host", "content-length"]
    }
    if backend_api_key:
        headers["Authorization"] = f"Bearer {backend_api_key}"
    try:
        response = await client.post(
            backend_url, params=request.query_params, json=json_body, headers=headers
        )
        return Response(
            content=response.content,
            media_type=response.headers.get("Content-Type"),
            status_code=response.status_code,
        )

    except httpx.ConnectError as e:
        logger.error(f"Connection error to backend {backend_url}: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")
    except httpx.ReadTimeout as e:
        logger.error(f"Read timeout from backend {backend_url}: {e}")
        raise HTTPException(status_code=504, detail="Backend request timed out")
    except Exception as e:
        logger.error(f"An error occurred during non-streaming proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {e}")


@app.get("/v1/models", summary="List available models")
async def list_models():
    """
    OpenAI 兼容接口: 列出所有可用的模型。
    """
    try:
        all_routes: list[ModelRoute] = await run_in_threadpool(get_all_routes_sync)

        models_data = []
        for route in all_routes:
            created_timestamp = int(route.created.timestamp())

            models_data.append(
                {
                    "id": route.model_name,
                    "object": "model",
                    # 3. 使用数据库中的 created 时间戳
                    "created": created_timestamp,
                    "owned_by": "openai_router",
                    "permission": [],
                }
            )

        response_data = {"object": "list", "data": models_data}

        logger.info(f"Returning {len(all_routes)} available models for /v1/models.")

        return response_data

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error when retrieving models: {e}"
        )


@app.post("/v1/responses", summary="/v1/responses ")
@app.post("/v1/completions", summary="/v1/completions")
@app.post("/v1/chat/completions", summary="/v1/chat/completions")
@app.post("/v1/embeddings", summary="/v1/embeddings")
@app.post("/v1/moderations", summary="/v1/moderations")
@app.post("/v1/images/generations", summary="/v1/images/generations")
@app.post("/v1/images/edits", summary="/v1/images/edits")
@app.post("/v1/images/variations", summary="/v1/images/variations")
@app.post("/v1/audio/transcriptions", summary="/v1/audio/transcriptions")
@app.post("/v1/audio/speech", summary="/v1/audio/speech")
@app.post("/v1/rerank", summary="/v1/rerank")
async def router(request: Request):
    backend_url, json_body, backend_api_key = await _get_routing_info(request)
    if json_body.get("stream", False):
        return StreamingResponse(
            _stream_proxy(backend_url, request, json_body, backend_api_key),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_proxy(backend_url, request, json_body, backend_api_key)


# --- Gradio 管理界面逻辑 (同步数据库操作) ---


def get_current_routes_sync():
    """同步获取当前路由表"""
    with Session(engine) as session:
        statement = select(ModelRoute)
        routes_db = session.exec(statement)
        routes = []
        for route in routes_db.all():
            # 屏蔽 API 密钥，仅显示最后 4 位
            masked_key = f"***{route.api_key[-4:]}" if route.api_key else "N/A (将透传)"
            routes.append([route.model_name, route.model_url, masked_key])
    return routes


def add_or_update_route_sync(model_name: str, model_url: str, api_key: str | None):
    """同步添加或更新一个路由到数据库"""
    status_message = ""
    # 将 Gradio 传来的空字符串视为空值
    if api_key is not None and not api_key.strip():
        api_key = None

    with Session(engine) as session:
        db_route = session.get(ModelRoute, model_name)

        if db_route:
            db_route.model_url = model_url
            db_route.api_key = api_key  # 更新 API 密钥
            status_message = f"路由 '{model_name}' 已更新。"
        else:
            db_route = ModelRoute(
                model_name=model_name, model_url=model_url, api_key=api_key
            )
            status_message = f"路由 '{model_name}' 已添加。"

        session.add(db_route)
        session.commit()

    logger.info(f"[Admin] {status_message}")
    return status_message


def delete_route_sync(model_name: str):
    """同步从数据库删除一个路由"""
    status_message = ""
    with Session(engine) as session:
        db_route = session.get(ModelRoute, model_name)

        if db_route:
            session.delete(db_route)
            session.commit()
            status_message = f"路由 '{model_name}' 已删除。"
            logger.info(f"[Admin] Route deleted: {model_name}")
        else:
            status_message = f"错误: 未找到路由 '{model_name}'。"

    return status_message


async def get_current_routes():
    """异步调用同步函数获取当前路由表"""
    return await run_in_threadpool(get_current_routes_sync)


async def add_or_update_route(model_name: str, model_url: str, api_key: str | None):
    """异步调用同步函数添加或更新路由"""
    if not model_name or not model_url:
        return "模型名称和 URL 不能为空", await get_current_routes()

    status_message = await run_in_threadpool(
        add_or_update_route_sync, model_name, model_url, api_key
    )
    return status_message, await get_current_routes()


async def delete_route(model_name: str):
    """异步调用同步函数删除路由"""
    if not model_name:
        return "要删除的模型名称不能为空", await get_current_routes()

    status_message = await run_in_threadpool(delete_route_sync, model_name)
    return status_message, await get_current_routes()


def on_select_route(routes_data: pd.DataFrame, evt: gr.SelectData):
    """
    Gradio: 当用户点击表格中的一行时，填充输入框。
    """
    if evt.index is None:
        return "", "", ""

    selected_row = routes_data.iloc[evt.index[0]]
    model_name = selected_row.iloc[0]
    model_url = selected_row.iloc[1]

    # 出于安全考虑，不回填 API 密钥
    # 用户必须在想要更新时手动重新输入
    return model_name, model_url, ""


def create_admin_ui():
    """创建 Gradio Blocks 界面"""
    with gr.Blocks(
        title="模型路由管理器", css="footer {display: none !important}"
    ) as admin_ui:
        gr.Markdown(
            "<h1 style='text-align:center;'>模型路由管理器</h1>", elem_id="title"
        )
        gr.Markdown(
            """**将不同端口、不同服务的`openAI`的接口通过统一的url进行路由！兼容 `vLLM`、`SGLang`、`lmdeoply`、`Ollama`等。**\n
**注意：** 所有路由配置都持久化到 `routes.db` 数据库中。您需要手动添加初始路由。"""
        )

        with gr.Row():
            refresh_button = gr.Button("刷新路由列表")

        with gr.Row():
            with gr.Column(scale=2):
                routes_datagrid = gr.DataFrame(
                    headers=[
                        "模型名称 (Model Name)",
                        "后端 URL (Backend URL)",
                        "API 密钥 (API Key)",
                    ],
                    label="当前路由表",
                    row_count=(1, "fixed"),
                    col_count=(3, "fixed"),
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 管理路由")
                status_output = gr.Textbox(
                    label="操作状态",
                    interactive=False,
                    value="这里用于显示上一次的操作状态",
                )
                model_name_input = gr.Textbox(label="模型名称", value="gpt4")
                model_url_input = gr.Textbox(
                    label="后端 URL", value="http://localhost:8082"
                )
                api_key_input = gr.Textbox(
                    label="后端 API 密钥 (可选)",
                    info="如果提供，路由器将使用此密钥覆盖原始请求中的 Authorization 标头。如果留空，将透传原始请求的密钥。",
                    type="password",  # 确保密钥在 UI 上被遮罩
                )
                with gr.Row():
                    add_update_button = gr.Button("添加 / 更新")
                    delete_button = gr.Button(
                        "删除",
                        variant="stop",
                    )

        # --- 绑定 Gradio 事件 ---
        admin_ui.load(get_current_routes, outputs=routes_datagrid)
        refresh_button.click(get_current_routes, outputs=routes_datagrid)

        add_update_button.click(
            add_or_update_route,
            inputs=[model_name_input, model_url_input, api_key_input],
            outputs=[status_output, routes_datagrid],
        )

        delete_button.click(
            delete_route,
            inputs=[model_name_input],
            outputs=[status_output, routes_datagrid],
        )

        routes_datagrid.select(
            on_select_route,
            inputs=[routes_datagrid],
            outputs=[model_name_input, model_url_input, api_key_input],
        )

    return admin_ui


# --- 挂载 Gradio 应用 ---
admin_interface = create_admin_ui()
app = gr.mount_gradio_app(app, admin_interface, path="/")

cli_app = typer.Typer()


@cli_app.command()
def main(
    host: Annotated[
        str, typer.Option(help="指定监听的主机地址", show_default=True)
    ] = "localhost",
    port: Annotated[
        int, typer.Option(help="指定监听的主机端口", show_default=True)
    ] = 8000,
):
    base_url = f"http://{host}:{port}"
    logger.info(f"UI 界面: http://{host}:{port}")
    logger.info(f"openAI API 文档: http://{host}:{port}/docs")
    time.sleep(1)
    try:
        if "0.0.0.0" in base_url:
            base_url = f"http://localhost:{port}"
        webbrowser.open_new_tab(base_url)
    except Exception as e:
        logger.warning(f"无法自动打开浏览器: {e}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli_app()
