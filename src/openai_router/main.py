import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional, Sequence

import gradio as gr
import httpx
import pandas as pd
import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from sqlalchemy.engine import Engine  # 导入同步 Engine
# --- 导入 SQLModel 和同步组件 ---
# 切换到同步 Session 和 Engine
from sqlmodel import Field, SQLModel, create_engine, Session, select
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).absolute().parent.parent.parent
SQLITE_DB_DIR = BASE_DIR / "data"  # 确保 data 目录存在
SQLITE_DB_DIR.mkdir(exist_ok=True)  # 自动创建
# --- 数据库配置 ---
SQLITE_DB_FILE = str(SQLITE_DB_DIR / "routes.db")
# 使用同步 SQLite URL
SQLITE_URL = f"sqlite:///{SQLITE_DB_FILE}"

# --- 全局变量 ---
client: Optional[httpx.AsyncClient] = None
# 同步 Engine
engine: Optional[Engine] = None

# 1. 存储每个 model_name 的下一个索引
#    defaultdict(int) 会在键不存在时自动创建并赋值为 0
model_rr_counters = defaultdict(int)
# 2. 为每个 model_name 提供一个独立的锁，以防止并发访问 rr_counters 时的竞争条件
#    defaultdict(threading.Lock) 会在键不存在时自动创建一个新的锁
model_rr_locks = defaultdict(threading.Lock)


# --- SQLModel 数据模型 ---
class ModelRoute(SQLModel, table=True):
    """
    一个 SQLModel 模型，用于存储模型名称到后端 URL 的路由。
    'id' 是主键，允许 'model_name' 重复以实现负载均衡。
    """

    # <--- MODIFIED: 架构变更 --->
    # 1. 'model_name' 不再是主键。
    # 2. 我们添加了一个自动递增的 'id' 作为主键。
    id: Optional[int] = Field(default=None, primary_key=True)
    model_name: str = Field(index=True)  # 仍然索引 'model_name' 以提高查询速度
    model_url: str
    api_key: str | None = Field(default=None)
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )


class RoutePath(StrEnum):
    COMPLETIONS = "/v1/completions"
    CHAT_COMPLETIONS = "/v1/chat/completions"
    EMBEDDINGS = "/v1/embeddings"
    IMAGES_GENERATIONS = "/v1/images/generations"
    IMAGES_EDITS = "/v1/images/edits"
    IMAGES_VARIATIONS = "/v1/images/variations"
    AUDIO_TRANSCRIPTIONS = "/v1/audio/transcriptions"
    AUDIO_TRANSLATIONS = "/v1/audio/translations"
    RERANK = "/v1/rerank"
    TOKENIZE = "/tokenize"
    DETOKENIZE = "/detokenize"


class DefaultRoute(SQLModel, table=True):
    """
    a SQLModel for staring default model of route
    """
    route: RoutePath = Field(primary_key=True, index=True)
    default_model: str


# --- 同步数据库辅助函数 (将在线程池中执行) ---


def create_db_and_tables_sync():
    """同步创建数据库和表"""
    SQLModel.metadata.create_all(engine)


def get_all_model_names_sync() -> Sequence[str]:
    """同步地从数据库中查询所有可用的模型名称"""
    with Session(engine) as session:
        statement = select(ModelRoute.model_name)
        results = session.exec(statement)
        available_models = results.all()
    return available_models


def get_all_routes_sync() -> Sequence[ModelRoute]:
    """同步地从数据库中查询所有可用的 ModelRoute 记录"""
    with Session(engine) as session:
        # 查询 ModelRoute 的所有记录
        statement = select(ModelRoute)
        results = session.exec(statement)
        # 返回 ModelRoute 对象的列表
        all_routes = results.all()
    return all_routes


def get_routing_info_sync(model: str) -> tuple[Optional[str], Sequence[str], Optional[str]]:
    """同步地从数据库中查询模型和所有可用模型"""
    with Session(engine) as session:
        # <--- MODIFIED: 核心负载均衡逻辑 --->

        # 1. 获取 *所有* 匹配该模型的路由
        statement = select(ModelRoute).where(ModelRoute.model_name == model)
        # .all() 将所有匹配的路由加载到列表中
        db_routes_list: Sequence[ModelRoute] = session.exec(statement).all()

        # 2. 获取所有可用模型名称 (用于错误消息)
        available_models = get_all_model_names_sync()

        if not db_routes_list:
            # 3. 如果列表为空，则未找到模型
            server = None
            backend_api_key = None
        elif len(db_routes_list) == 1:
            # 4. 优化：如果只有一个，直接返回，无需加锁或计数
            selected_route = db_routes_list[0]
            server = selected_route.model_url
            backend_api_key = selected_route.api_key
        else:
            # 5. 轮询 (Round-Robin) 逻辑
            # 获取此模型的特定锁
            lock = model_rr_locks[model]

            # 使用 'with' 语句自动获取和释放锁（线程安全）
            with lock:
                # 5.1. 获取当前计数器值 (默认为 0)
                current_index = model_rr_counters[model]

                # 5.2. 计算下一个索引，使用模运算实现循环
                next_index = (current_index + 1) % len(db_routes_list)

                # 5.3. 更新全局计数器，为 *下一个* 请求做准备
                model_rr_counters[model] = next_index

            # 5.4. 使用我们在此次请求中获取的 current_index 来选择路由
            # （在锁之外执行，因为我们已经拿到了索引）
            selected_route = db_routes_list[current_index]

            server = selected_route.model_url
            backend_api_key = selected_route.api_key

            logger.debug(
                f"Round-Robin: model={model}, selected_index={current_index}, total={len(db_routes_list)}"
            )

        return server, available_models, backend_api_key

def get_default_model_route_info_sync(path: str) -> tuple[Optional[str], Sequence[str], Optional[str], Optional[str]]:
    """同步地从数据库中查询默认模型路由信息"""
    with Session(engine) as session:
        statement = select(DefaultRoute.default_model).where(DefaultRoute.route == path)
        model: Optional[str] = session.exec(statement).first()

        if model:
            return get_routing_info_sync(model) + (model,)
        else:
            return None, get_all_model_names_sync(), None, None

# --- FastAPI 生命周期 ---

import os
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError, DBAPIError


@asynccontextmanager
async def lifespan(_: FastAPI):
    global client, engine

    # --- 启动时的逻辑 ---

    # 1. 初始化数据库引擎 (同步)
    temp_engine = create_engine(SQLITE_URL, echo=False)

    models_to_check = [ModelRoute, DefaultRoute]
    # ---------- 检查数据库 schema 是否与 SQLModel 定义匹配 ----------
    try:
        # 1. 获取 Inspector
        inspector = inspect(temp_engine)

        schema_mismatch = False

        for model in models_to_check:
            table_name = model.__tablename__

            # 使用 has_table 先检查表是否存在，避免直接 get_columns 抛出异常
            # 这样可以允许部分表存在(旧)，部分表不存在(新加的)的情况
            if inspector.has_table(table_name):
                # 2. 获取数据库中表的实际列
                actual_columns_info = inspector.get_columns(table_name)
                actual_columns = {col["name"] for col in actual_columns_info}

                # 3. 获取 SQLModel 中定义的预期列
                expected_columns = {col.name for col in model.__table__.columns}

                # 4. 比较
                if actual_columns != expected_columns:
                    logger.warning(
                        f"数据库表 '{table_name}' Schema 不匹配! \n预期: {expected_columns}\n实际: {actual_columns}"
                    )
                    schema_mismatch = True
                    break # 只要有一个表不对，就触发重置
            else:
                # 表不存在，这很正常（可能是新加的功能），不需要删除数据库
                # 后续的 create_db_and_tables_sync 会自动创建它
                logger.info(f"表 '{table_name}' 尚未存在，将在后续步骤创建。")

        # 如果发现任何 Schema 不匹配，执行删除重建逻辑
        if schema_mismatch:
            logger.warning(f"正在删除旧数据库文件: {SQLITE_DB_FILE} 以进行重建...")

            temp_engine.dispose()

            try:
                os.remove(SQLITE_DB_FILE)
                logger.info(f"成功删除旧数据库: {SQLITE_DB_FILE}")
            except OSError as e:
                logger.error(
                    f"无法删除数据库文件 {SQLITE_DB_FILE}: {e}. 请手动删除它。"
                )

            # 重新创建 engine 实例
            temp_engine = create_engine(SQLITE_URL, echo=False)


    except (OperationalError, DBAPIError):
        # 这是预期的“错误”，如果数据库文件或 'modelroute' 表还不存在
        # 我们安全地忽略它，然后继续创建表
        logger.info(f"数据库或表 '{ModelRoute.__tablename__}' 未找到。将创建新表。")
    except Exception as e:
        # 捕获任何其他意外的检查错误
        logger.error(f"数据库 schema 检查期间发生意外错误: {e}")
        logger.warning("将尝试继续，但可能会失败。")
        import traceback

        traceback.print_exc()
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
        server, available_models, backend_api_key, model = await run_in_threadpool(
            get_default_model_route_info_sync, request.url.path
        )

        if server is None:
            raise HTTPException(
                status_code=400,
                detail=f"Can't find default model for path {request.url.path}. Available models: {available_models}",
            )

        json_body["model"] = model  # 设置默认模型到请求体中
        logger.info(f"Using default model '{model}' for path {request.url.path}")
    else:
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
        response = await client.request(
            request.method, backend_url, params=request.query_params, json=json_body, headers=headers
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
    (此函数必须修改以处理重复的模型名称)
    """
    try:

        # 1. 获取所有路由条目
        all_routes: Sequence[ModelRoute] = await run_in_threadpool(get_all_routes_sync)

        # 2. 按 model_name 分组，并找到每组中 *最早* 的 created 时间戳
        models_by_name = {}  # { model_name: earliest_timestamp }
        for route in all_routes:
            created_timestamp = int(route.created.timestamp())

            if route.model_name not in models_by_name:
                models_by_name[route.model_name] = created_timestamp
            else:
                # 更新为更早（更小）的时间戳
                models_by_name[route.model_name] = min(
                    models_by_name[route.model_name], created_timestamp
                )

        # 3. 构建唯一的模型数据列表
        models_data = []
        for model_name, created_timestamp in models_by_name.items():
            models_data.append(
                {
                    "id": model_name,
                    "object": "model",
                    "created": created_timestamp,
                    "owned_by": "openai_router",
                    "permission": [],
                }
            )

        response_data = {"object": "list", "data": models_data}

        logger.info(
            f"Returning {len(models_data)} unique available models for /v1/models."
        )

        return response_data

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error when retrieving models: {e}"
        )


@app.post(RoutePath.COMPLETIONS, summary=RoutePath.COMPLETIONS)
@app.post(RoutePath.CHAT_COMPLETIONS, summary=RoutePath.CHAT_COMPLETIONS)
@app.post(RoutePath.EMBEDDINGS, summary=RoutePath.EMBEDDINGS)
@app.post(RoutePath.IMAGES_GENERATIONS, summary=RoutePath.IMAGES_GENERATIONS)
@app.post(RoutePath.IMAGES_EDITS, summary=RoutePath.IMAGES_EDITS)
@app.post(RoutePath.IMAGES_VARIATIONS, summary=RoutePath.IMAGES_VARIATIONS)
@app.post(RoutePath.AUDIO_TRANSCRIPTIONS, summary=RoutePath.AUDIO_TRANSCRIPTIONS)
@app.post(RoutePath.AUDIO_TRANSLATIONS, summary=RoutePath.AUDIO_TRANSLATIONS)
@app.post(RoutePath.RERANK, summary=RoutePath.RERANK)
@app.post(RoutePath.TOKENIZE, summary=RoutePath.TOKENIZE)
@app.post(RoutePath.DETOKENIZE, summary=RoutePath.DETOKENIZE)
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


def get_current_routes_sync() -> list[tuple[str, str, str]]:
    """同步获取当前路由表"""
    with Session(engine) as session:
        statement = select(ModelRoute)
        routes_db = session.exec(statement)
        routes: list[tuple[str, str, str]] = []
        for route in routes_db.all():
            # 屏蔽 API 密钥，仅显示最后 4 位
            masked_key = f"***{route.api_key[-4:]}" if route.api_key else "N/A (将透传)"
            routes.append((route.model_name, route.model_url, masked_key))
    return routes


def get_current_default_routes_sync() -> list[tuple[str, str]]:
    """同步获取当前預設路由表"""
    with Session(engine) as session:
        statement = select(DefaultRoute)
        routes_db = session.exec(statement)
        routes: list[tuple[str, str]] = []
        for route in routes_db.all():
            routes.append((route.route, route.default_model))
    return routes


def add_or_update_route_sync(model_name: str, model_url: str, api_key: str | None):
    """同步添加或更新一个路由到数据库"""
    status_message = ""
    # 将 Gradio 传来的空字符串视为空值
    if api_key is not None and not api_key.strip():
        api_key = None

    with Session(engine) as session:
        statement = select(ModelRoute).where(
            ModelRoute.model_name == model_name, ModelRoute.model_url == model_url
        )
        db_route = session.exec(statement).first()  # 使用 .first()

        if db_route:
            # 2. 如果存在，我们只更新 API 密钥
            db_route.api_key = api_key
            status_message = f"路由 '{model_name} -> {model_url}' 的 API 密钥已更新。"
        else:
            # 3. 如果不存在，我们创建一个全新的条目
            db_route = ModelRoute(
                model_name=model_name, model_url=model_url, api_key=api_key
            )
            status_message = (
                f"新路由 '{model_name} -> {model_url}' 已添加 (用于负载均衡)。"
            )

        session.add(db_route)
        session.commit()

    logger.info(f"[Admin] {status_message}")
    return status_message


def add_or_update_default_route_sync(route: str, default_model: str):
    """同步添加或更新一个預設路由到数据库"""
    status_message = ""

    with Session(engine) as session:
        statement = select(DefaultRoute).where(
            DefaultRoute.route == route
        )
        db_route = session.exec(statement).first()  # 使用 .first()

        if db_route:
            # 2. 如果存在，我们只更新默认模型
            db_route.default_model = default_model
            status_message = f"預設路由 '{route}' 的默认模型已更新为 '{default_model}'。"
        else:
            # 3. 如果不存在，我们创建一个全新的条目
            db_route = DefaultRoute(
                route=route, default_model=default_model
            )
            status_message = (
                f"新預設路由 '{route} -> {default_model}' 已添加。"
            )

        session.add(db_route)
        session.commit()

    logger.info(f"[Admin] {status_message}")
    return status_message


def delete_route_sync(model_name: str, model_url: str):  # <--- MODIFIED: 需要 model_url
    """
    同步从数据库删除一个 *特定* 的 (model_name, model_url) 路由。
    """
    status_message = ""
    with Session(engine) as session:
        # <--- MODIFIED: 删除逻辑 --->
        # 1. 寻找一个 (model_name, model_url) 组合的条目
        statement = select(ModelRoute).where(
            ModelRoute.model_name == model_name, ModelRoute.model_url == model_url
        )
        db_route = session.exec(statement).first()
        # <--- END MODIFIED --->

        if db_route:
            session.delete(db_route)
            session.commit()
            status_message = f"路由 '{model_name} -> {model_url}' 已删除。"
            logger.info(f"[Admin] Route deleted: {model_name} -> {model_url}")
        else:
            status_message = f"错误: 未找到路由 '{model_name} -> {model_url}'。"

    return status_message


def delete_default_route_sync(path: str):
    """
    同步从数据库删除一个預設路由。
    """
    status_message = ""
    with Session(engine) as session:
        statement = select(DefaultRoute).where(
            DefaultRoute.route == path
        )
        db_route = session.exec(statement).first()

        if db_route:
            session.delete(db_route)
            session.commit()
            status_message = f"預設路由 '{path}' 已删除。"
            logger.info(f"[Admin] Default route deleted: {path}")
        else:
            status_message = f"错误: 未找到預設路由 '{path}'。"

    return status_message


async def get_current_routes():
    """异步调用同步函数获取当前路由表"""
    return await run_in_threadpool(get_current_routes_sync)


async def get_current_default_routes():
    """异步调用同步函数获取当前預設路由表"""
    return await run_in_threadpool(get_current_default_routes_sync)


async def add_or_update_route(model_name: str, model_url: str, api_key: str | None):
    """异步调用同步函数添加或更新路由"""
    if not model_name or not model_url:
        return "模型名称和 URL 不能为空", await get_current_routes()

    status_message = await run_in_threadpool(
        add_or_update_route_sync, model_name, model_url, api_key
    )
    return status_message, await get_current_routes()


async def add_or_update_default_route(route: str, default_model: str):
    """异步调用同步函数添加或更新預設路由"""
    if not route or not default_model:
        return "路由和預設模型不能为空", await get_current_default_routes()

    status_message = await run_in_threadpool(
        add_or_update_default_route_sync, route, default_model
    )
    return status_message, await get_current_default_routes()


async def delete_route(
    model_name: str, model_url: str
):  # <--- MODIFIED: 需要 model_url
    """异步调用同步函数删除路由"""
    if not model_name or not model_url:
        return "要删除的模型名称和 URL 均不能为空", await get_current_routes()

    status_message = await run_in_threadpool(delete_route_sync, model_name, model_url)
    return status_message, await get_current_routes()


async def delete_default_route(path: str):
    """异步调用同步函数删除預設路由"""
    if not path:
        return "要删除的路由不能为空", await get_current_default_routes()

    status_message = await run_in_threadpool(delete_default_route_sync, path)
    return status_message, await get_current_default_routes()


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


def on_select_default_route(routes_data: pd.DataFrame, evt: gr.SelectData):
    """
    Gradio: 当用户点击预设路由表格中的一行时，填充输入框。
    """
    if evt.index is None:
        return "", ""

    selected_row = routes_data.iloc[evt.index[0]]
    route = selected_row.iloc[0]
    default_model = selected_row.iloc[1]

    return route, default_model


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

        with gr.Tab("路由管理"):
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
                        label="当前路由表 (同一模型可有多个URL)",
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
                            "删除 (指定URL)",
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
                inputs=[model_name_input, model_url_input],
                outputs=[status_output, routes_datagrid],
            )

            routes_datagrid.select(
                on_select_route,
                inputs=[routes_datagrid],
                outputs=[model_name_input, model_url_input, api_key_input],
            )
        with gr.Tab("預設路由管理"):
            with gr.Row():
                refresh_button = gr.Button("刷新路由列表")

            with gr.Row():
                with gr.Column(scale=2):
                    routes_datagrid = gr.DataFrame(
                        headers=[
                            "路由 (Route)",
                            "預設模型 (Default Model)",
                        ],
                        label="當前預設路由表",
                        row_count=(1, "fixed"),
                        col_count=(2, "fixed"),
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 管理預設路由")
                    status_output = gr.Textbox(
                        label="操作狀態",
                        interactive=False,
                        value="這裡用於顯示上一次的操作狀態",
                    )
                    route_input = gr.Dropdown(list(RoutePath._value2member_map_.keys()), label="路由", value="/v1/chat/completions")
                    default_model_input = gr.Textbox(
                        label="預設模型", value="gpt4"
                    )
                    with gr.Row():
                        add_update_button = gr.Button("添加 / 更新")
                        delete_button = gr.Button(
                            "刪除 (指定路由)",
                            variant="stop",
                        )

            # --- 绑定 Gradio 事件 ---
            admin_ui.load(get_current_default_routes, outputs=routes_datagrid)
            refresh_button.click(get_current_default_routes, outputs=routes_datagrid)

            add_update_button.click(
                add_or_update_default_route,
                inputs=[route_input, default_model_input],
                outputs=[status_output, routes_datagrid],
            )

            delete_button.click(
                delete_default_route,
                inputs=[route_input],
                outputs=[status_output, routes_datagrid],
            )

            routes_datagrid.select(
                on_select_default_route,
                inputs=[routes_datagrid],
                outputs=[route_input, default_model_input],
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
    logger.info(f"UI 界面: {base_url}")
    logger.info(f"openAI API 文档: {base_url}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli_app()
