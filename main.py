from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Request, status as http_status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sqlite3
import uuid
import datetime
import json
import os
import shutil
from pydantic import BaseModel, Field, Extra
from typing import List, Dict, Any, Optional, AsyncIterable
from pathlib import Path
import re
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import openai 

# --- 配置 (Configuration) ---
DATABASE_URL = "./chat_app.db"
UPLOAD_DIRECTORY = "./inputs" 
RAG_SERVICE_BASE_URL = "http://localhost:9621"

# LLM 配置:(按需修改成你需要的LLM)
# TODO
LLM_CONFIGS = {
    "deepseek/deepseek-v3-base:free": {
        "type": "openai",
        "model_name": "",
        "api_key": "", 
        "base_url": "https://openrouter.ai/api/v1",     
    },
    "deepseek/deepseek-v3-base:free(finetuned)": {
        "type": "openai",
        "model_name": "",
        "api_key": "", 
        "base_url": "https://openrouter.ai/api/v1",     
    },
}

app = FastAPI(title="Chat Application Backend")

# --- CORS 中间件 (CORS Middleware) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
URL_REGEX = re.compile(r"https?://[^\s]+")

async def fetch_markdown(url: str, depth: int = 1, max_len: int = 4_000) -> str:
    """
    抓指定 URL（仅一层内部链接）并返回 Markdown 文本。
    适当裁剪，避免 prompt 过长。
    """
    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        visited = set()
        collected = []

        async def _crawl(u: str, d: int):
            if u in visited or "#" in u:
                return
            visited.add(u)
            res = await crawler.arun(url=u, config=run_conf)
            if res and res.markdown:
                collected.append(f"\n---  {u}  ---\n{res.markdown}")
                if d == 1:   # 只递归一层
                    for link in res.links.get("internal", []):
                        href = link.get("href", "")
                        if href.startswith("http"):
                            await _crawl(href, d + 1)

        await _crawl(url, 1)

    md_text = "\n".join(collected)
    if len(md_text) > max_len:
        md_text = md_text[:max_len] + "\n\n...(已截断)"
    return md_text
# --- 数据库设置 (Database Setup) ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        uuid TEXT PRIMARY KEY, title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ) """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_uuid TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        order_id INTEGER NOT NULL,
        FOREIGN KEY (conversation_uuid) REFERENCES conversations (uuid) ON DELETE CASCADE )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL UNIQUE,
        filepath TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        size INTEGER,
        file_type TEXT
    )""")
    conn.commit()
    conn.close()

create_tables()
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- Pydantic 模型 (Pydantic Models) ---
class ConversationCreateResponse(BaseModel):
    uuid: str; title: str; created_at: datetime.datetime

class ConversationListItem(BaseModel):
    key: str = Field(..., alias="uuid"); label: str = Field(..., alias="title"); group: str

class MessageBase(BaseModel):
    role: str; content: str

class MessageAPIResponse(MessageBase):
    id: int; timestamp: datetime.datetime; order_id: int

class ChatPayload(BaseModel):
    messages: List[MessageBase]
    model: str
    stream: Optional[bool] = True
    class Config:
        extra = Extra.allow

class RenameConversationRequest(BaseModel):
    title: str

class DocumentInfo(BaseModel):
    id: Optional[int] = None
    filename: str
    filepath: str
    uploaded_at: datetime.datetime
    size: Optional[int] = None
    file_type: Optional[str] = None


# --- 辅助函数 (Helper Functions) ---
async def store_message(conversation_uuid: str, role: str, content: str, conn) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(order_id) FROM messages WHERE conversation_uuid = ?", (conversation_uuid,))
    max_order_id_row = cursor.fetchone()
    next_order_id = (max_order_id_row[0] if max_order_id_row and max_order_id_row[0] is not None else 0) + 1

    content_to_store = content

    cursor.execute(
        "INSERT INTO messages (conversation_uuid, role, content, order_id) VALUES (?, ?, ?, ?)",
        (conversation_uuid, role, content_to_store, next_order_id))

    if role == 'user' and next_order_id == 1:
        cursor.execute("SELECT title FROM conversations WHERE uuid = ?", (conversation_uuid,))
        current_title_row = cursor.fetchone()
        if current_title_row and current_title_row['title'] and current_title_row['title'].startswith("新会话"):
            new_title = content_to_store[:30].strip() + "..." if len(content_to_store) > 30 else content_to_store.strip()
            if not new_title:
                pass
            else:
                cursor.execute("UPDATE conversations SET title = ? WHERE uuid = ?", (new_title, conversation_uuid))

    cursor.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE uuid = ?", (conversation_uuid,))
    conn.commit()
    return next_order_id

# --- API 端点 (API Endpoints) ---
@app.post("/conversations", response_model=ConversationCreateResponse, status_code=http_status.HTTP_201_CREATED)
async def create_conversation_endpoint():
    conn = get_db_connection(); cursor = conn.cursor()
    new_uuid = str(uuid.uuid4())
    cursor.execute("SELECT COUNT(*) FROM conversations")
    count = cursor.fetchone()[0]
    title = f"新会话 {count + 1}"

    try:
        cursor.execute("INSERT INTO conversations (uuid, title) VALUES (?, ?)", (new_uuid, title))
        conn.commit()
        created_conv_row = cursor.execute("SELECT uuid, title, created_at FROM conversations WHERE uuid = ?", (new_uuid,)).fetchone()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        if conn: conn.close()
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"数据库错误: {e}")

    if conn: conn.close()

    if not created_conv_row:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="创建或检索会话失败")

    return ConversationCreateResponse(
        uuid=created_conv_row["uuid"],
        title=created_conv_row["title"],
        created_at=created_conv_row["created_at"]
    )

@app.get("/conversations", response_model=List[ConversationListItem])
async def list_conversations_endpoint():
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT uuid, title, updated_at FROM conversations ORDER BY updated_at DESC")
    conversations_db = cursor.fetchall(); conn.close()
    formatted_conversations = []
    today_date = datetime.date.today(); yesterday_date = today_date - datetime.timedelta(days=1)
    for conv_row in conversations_db:
        updated_at_dt = datetime.datetime.fromisoformat(conv_row["updated_at"].split('.')[0]) if isinstance(conv_row["updated_at"], str) else conv_row["updated_at"]
        conv_date = updated_at_dt.date()
        group = "更早"
        if conv_date == today_date: group = "今天"
        elif conv_date == yesterday_date: group = "昨天"
        title_to_send = conv_row["title"].strip() if conv_row["title"] and conv_row["title"].strip() else f"会话 {conv_row['uuid'][:8]}"
        formatted_conversations.append(ConversationListItem(uuid=conv_row["uuid"], title=title_to_send, group=group))
    return formatted_conversations

@app.get("/conversations/{conversation_uuid}/messages", response_model=List[MessageAPIResponse])
async def get_messages_endpoint(conversation_uuid: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT id, role, content, timestamp, order_id FROM messages WHERE conversation_uuid = ? ORDER BY order_id ASC", (conversation_uuid,))
    messages_db = cursor.fetchall(); conn.close()
    return [MessageAPIResponse(id=m["id"], role=m["role"], content=m["content"], timestamp=m["timestamp"], order_id=m["order_id"]) for m in messages_db]

@app.delete("/conversations/{conversation_uuid}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_conversation_endpoint(conversation_uuid: str):
    if not conversation_uuid or not isinstance(conversation_uuid, str) or len(conversation_uuid) < 10:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="无效的会话 ID。")
    conn = get_db_connection(); cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE uuid = ?", (conversation_uuid,))
        if cursor.fetchone()[0] == 0: raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="会话未找到")
        cursor.execute("DELETE FROM conversations WHERE uuid = ?", (conversation_uuid,)); conn.commit()
    except HTTPException: raise
    except Exception as e: conn.rollback(); raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除会话时出错: {str(e)}")
    finally:
        if conn: conn.close()

@app.put("/conversations/{conversation_uuid}/rename", status_code=http_status.HTTP_200_OK)
async def rename_conversation_endpoint(conversation_uuid: str, payload: RenameConversationRequest):
    if not conversation_uuid or not isinstance(conversation_uuid, str) or len(conversation_uuid) < 10:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="无效的会话 ID。")
    new_title = payload.title.strip()
    if not new_title: new_title = f"会话 {conversation_uuid[:8]}"

    conn = get_db_connection(); cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM conversations WHERE uuid = ?", (conversation_uuid,))
        if cursor.fetchone()[0] == 0: raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="会话未找到")

        cursor.execute("UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE uuid = ?", (new_title, conversation_uuid))
        conn.commit()
    except Exception as e: conn.rollback(); raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"重命名会话时出错: {str(e)}")
    finally:
        if conn: conn.close()
    return {"message": "会话已成功重命名", "newTitle": new_title}

async def stream_openai_response(api_key: str, base_url: Optional[str], model_name: str, llm_api_messages: list, stream_params: dict) -> AsyncIterable[str]:
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url if base_url else None)
    try:
        stream = await client.chat.completions.create(model=model_name, messages=llm_api_messages, stream=True, **stream_params) # type: ignore
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                sse_event = {"choices": [{"delta": {"role": "assistant", "content": content}}]}
                yield f"data: {json.dumps(sse_event)}\n\n"
    except openai.APIConnectionError as e: yield f"data: {json.dumps({'choices': [{'delta': {'content': f'无法连接到 OpenAI 服务: {str(e.__cause__ if e.__cause__ else e)}'}}]})}\n\n"
    except openai.RateLimitError: yield f"data: {json.dumps({'choices': [{'delta': {'content': 'OpenAI API 请求频率过高，请稍后再试。'}}]})}\n\n"
    except openai.AuthenticationError: yield f"data: {json.dumps({'choices': [{'delta': {'content': 'OpenAI API 密钥无效或权限不足。'}}]})}\n\n"
    except Exception as e: yield f"data: {json.dumps({'choices': [{'delta': {'content': f'与 OpenAI 通信时发生未知错误: {str(e)}'}}]})}\n\n"
    finally: yield f"data: [DONE]\n\n"

async def stream_generic_sse_response(url: str, api_key: Optional[str], llm_api_payload: dict, request_headers: Optional[Dict[str, str]]) -> AsyncIterable[str]:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **(request_headers or {})}
    if api_key: headers["Authorization"] = api_key # Assuming generic SSE might use Bearer or other token
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=llm_api_payload, headers=headers) as response:
                if response.status_code != 200:
                    error_body = await response.aread(); error_content = f"LLM 服务错误 ({response.status_code})。"
                    if response.status_code == 401: error_content = "LLM API 密钥无效或权限不足。"
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': error_content}}]})}\n\n"
                else:
                    async for line_bytes in response.aiter_bytes():
                        line = line_bytes.decode('utf-8').strip()
                        if line: yield f"{line}\n\n" # Pass through SSE as is
    except httpx.RequestError as e_req: yield f"data: {json.dumps({'choices': [{'delta': {'content': f'无法连接到 LLM 服务: {str(e_req)}'}}]})}\n\n"
    except Exception as e_stream: yield f"data: {json.dumps({'choices': [{'delta': {'content': f'后端 LLM 流错误: {str(e_stream)}'}}]})}\n\n"
    finally: yield f"data: [DONE]\n\n"

@app.post("/chat/{conversation_uuid}/send_message")
async def send_message_endpoint(conversation_uuid: str, payload: ChatPayload, fastapi_request: Request):
    if not conversation_uuid or not isinstance(conversation_uuid, str) or len(conversation_uuid) < 10:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="无效的会话 ID。")
    conn_main = get_db_connection()
    try:
        cursor = conn_main.cursor()
        cursor.execute("SELECT uuid FROM conversations WHERE uuid = ?", (conversation_uuid,))
        if not cursor.fetchone(): raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="会话未找到。")
        user_message_obj = payload.messages[-1]
        if user_message_obj.role != "user": raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="消息列表的最后一条应为用户消息。")

        user_query = user_message_obj.content
        await store_message(conversation_uuid, user_message_obj.role, user_query, conn_main)
        # ---- 网页抓取增强 ----
        url_match = URL_REGEX.search(user_query)
        if url_match:
            url = url_match.group(0)
            try:
                md = await fetch_markdown(url)
                user_query = f"以下是用户给出的网页内容（含一级内部链接），请结合回答：\n{md}\n\n用户问题：{user_query}"
            except Exception as e:
                print(f"抓取 {url} 失败: {e}")
        # 把可能修改后的 user_query 写回
        llm_api_messages = [*payload.messages]
        llm_api_messages[-1].content = user_query

        rag_context_str = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                rag_api_payload = {"query": user_query, "mode": "hybrid", "only_need_context": True}
                rag_response = await client.post(f"{RAG_SERVICE_BASE_URL}/query", json=rag_api_payload)
                rag_response.raise_for_status()
                rag_data = rag_response.json()
                rag_context_str = rag_data.get("response", "")
            except Exception as e:
                print(f"RAG 服务调用或处理失败: {e}")
        print(rag_context_str)
        llm_api_messages = [msg.model_dump() for msg in payload.messages]
        if rag_context_str:
            if llm_api_messages and llm_api_messages[-1]["role"] == "user":
                llm_api_messages[-1]["content"] = (f"请参考以下背景信息来回答用户的问题:\n---背景信息开始---\n{rag_context_str}\n---背景信息结束---\n\n用户问题: {llm_api_messages[-1]['content']}")
            else: llm_api_messages.insert(0, {"role": "system", "content": f"背景信息:\n{rag_context_str}"})

        model_config_key = payload.model; model_config = LLM_CONFIGS.get(model_config_key)

        if not model_config:
            async def err_gen_no_model_config():
                error_msg = f'错误: 模型配置键 "{model_config_key}" 未在后端找到。'
                yield f"data: {json.dumps({'choices':[{'delta':{'content': error_msg }}]})}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(err_gen_no_model_config(), media_type="text/event-stream")

        stream_params = {k: v for k, v in payload.model_extra.items() if k in ["temperature", "top_p", "max_tokens"]} if payload.model_extra else {}

        full_assistant_response_accumulator = ""
        async def response_wrapper_generator() -> AsyncIterable[str]:
            nonlocal full_assistant_response_accumulator; response_generator: AsyncIterable[str]
            if model_config["type"] == "openai":
                api_key = model_config.get("api_key")
                base_url = model_config.get("base_url") # Can be None or empty string

                if not api_key:
                    error_msg = f'错误: 模型 "{model_config_key}" 的 API 密钥未在配置中直接指定。'
                    yield f"data: {json.dumps({'choices':[{'delta':{'content': error_msg }}]})}\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                # Pass base_url as is; openai client handles None or empty string correctly
                response_generator = stream_openai_response(api_key, base_url, model_config["model_name"], llm_api_messages, stream_params)
            elif model_config["type"] == "generic_sse":
                url = os.getenv(model_config["url_env"])
                api_key = os.getenv(model_config["api_key_env"])
                if not url:
                    error_msg = f'错误: 模型 "{model_config_key}" 的 URL (环境变量 {model_config["url_env"]}) 未设置。'
                    yield f"data: {json.dumps({'choices':[{'delta':{'content': error_msg }}]})}\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                generic_payload = {"model": model_config.get("model_name", model_config_key), "messages": llm_api_messages, "stream": True, **stream_params}
                response_generator = stream_generic_sse_response(url, api_key, generic_payload, model_config.get("headers"))
            else:
                error_msg = f'错误: 模型 "{model_config_key}" 的类型 "{model_config["type"]}" 不支持。'
                yield f"data: {json.dumps({'choices':[{'delta':{'content': error_msg }}]})}\n\n"
                yield f"data: [DONE]\n\n"
                return

            async for sse_event_str in response_generator:
                yield sse_event_str
                if sse_event_str.startswith("data:") and not sse_event_str.strip().endswith("[DONE]"):
                    try:
                        data_content = json.loads(sse_event_str[len("data:"):].strip())
                        if data_content.get("choices") and data_content["choices"][0].get("delta"):
                            full_assistant_response_accumulator += data_content["choices"][0]["delta"].get("content", "")
                    except: pass
            if full_assistant_response_accumulator:
                conn_store = get_db_connection()
                try: await store_message(conversation_uuid, "assistant", full_assistant_response_accumulator, conn_store)
                finally: conn_store.close()
        return StreamingResponse(response_wrapper_generator(), media_type="text/event-stream")

    except HTTPException:
        if 'conn_main' in locals() and hasattr(conn_main, 'closed') and not conn_main.closed : conn_main.close()
        raise
    except Exception as e:
        if 'conn_main' in locals() and hasattr(conn_main, 'closed') and not conn_main.closed: conn_main.rollback(); conn_main.close()
        print(f"发送消息主处理流程错误: {e}")
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"发送消息失败: {str(e)}")
    finally:
        if 'conn_main' in locals() and hasattr(conn_main, 'closed') and not conn_main.closed:
            conn_main.close()

@app.post("/documents/upload", status_code=http_status.HTTP_202_ACCEPTED)
async def upload_document_endpoint(file: UploadFile = File(...)):
    filename = file.filename
    if not filename: raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="文件名不能为空。")
    if ".." in filename or filename.startswith("/"): raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="无效的文件名。")

    upload_dir_path = Path(UPLOAD_DIRECTORY)
    upload_dir_path.mkdir(parents=True, exist_ok=True)
    file_location = upload_dir_path / filename

    try:
        file_content = await file.read()
        with open(file_location, "wb+") as file_object:
            file_object.write(file_content)

        file_size = file_location.stat().st_size
        file_type = file.content_type or Path(filename).suffix

        db_conn_docs = get_db_connection(); cursor_docs = db_conn_docs.cursor()
        cursor_docs.execute(
            """
            INSERT INTO documents (filename, filepath, size, file_type) VALUES (?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                filepath=excluded.filepath,
                uploaded_at=CURRENT_TIMESTAMP,
                size=excluded.size,
                file_type=excluded.file_type
            """, (filename, str(file_location), file_size, file_type)
        )
        db_conn_docs.commit(); db_conn_docs.close()

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                scan_response = await client.post(f"{RAG_SERVICE_BASE_URL}/documents/scan", timeout=1800)
                scan_response.raise_for_status()
                return {"upload_status": "File uploaded and RAG scan initiated.", "scan_response": scan_response.json()}
            except Exception as e_scan:
                print(f"RAG scan initiation failed: {e_scan}")
                return {"upload_status": "File uploaded, but RAG scan initiation failed.", "error": str(e_scan)}

    except IOError as e_io: raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"文件系统错误: {e_io}")
    except Exception as e_gen:
        if file_location.exists():
            try: os.remove(file_location)
            except: pass
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"上传和处理文件失败: {str(e_gen)}")
    finally: await file.close()

@app.get("/documents", response_model=List[DocumentInfo])
async def list_local_documents_endpoint():
    db_conn = get_db_connection()
    cursor = db_conn.cursor()
    cursor.execute("SELECT id, filename, filepath, uploaded_at, size, file_type FROM documents ORDER BY uploaded_at DESC")
    db_documents = cursor.fetchall()
    db_conn.close()

    documents_info = []
    for doc_row in db_documents:
        if Path(doc_row["filepath"]).exists():
            documents_info.append(DocumentInfo(
                id=doc_row["id"], filename=doc_row["filename"], filepath=doc_row["filepath"],
                uploaded_at=doc_row["uploaded_at"], size=doc_row["size"], file_type=doc_row["file_type"]
            ))
        else:
            print(f"Warning: File {doc_row['filename']} listed in DB but not found at {doc_row['filepath']}")
    return documents_info

@app.delete("/documents/{filename}", status_code=http_status.HTTP_200_OK)
async def delete_local_document_endpoint(filename: str):
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="无效的文件名。")

    db_conn = get_db_connection(); cursor = db_conn.cursor() # Changed conn to db_conn to avoid conflict with outer scope
    try:
        cursor.execute("SELECT filepath FROM documents WHERE filename = ?", (filename,))
        db_entry = cursor.fetchone()
        if not db_entry: raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="数据库中未找到该文件记录。")
        actual_filepath_from_db = Path(db_entry["filepath"])
        if actual_filepath_from_db.exists():
            try: os.remove(actual_filepath_from_db)
            except OSError as e: print(f"删除文件时出错 {actual_filepath_from_db}: {e}")
        else: print(f"警告: 文件 {filename} 在数据库中，但在路径 {actual_filepath_from_db} 未找到。")
        cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,)); db_conn.commit()
        if cursor.rowcount == 0: raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="尝试从数据库删除文件记录失败。")
        return {"message": f"文件 '{filename}' 已成功删除。"}
    except HTTPException: raise
    except Exception as e: db_conn.rollback(); raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除文件时出错: {str(e)}")
    finally:
        if db_conn: db_conn.close() # Ensure db_conn is closed

@app.get("/documents/pipeline_status")
async def get_pipeline_status_endpoint():
    async with httpx.AsyncClient() as client:
        try: resp = await client.get(f"{RAG_SERVICE_BASE_URL}/documents/pipeline_status"); resp.raise_for_status(); return JSONResponse(content=resp.json())
        except httpx.RequestError as e: raise HTTPException(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"RAG 服务不可用 (流水线状态): {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404: return JSONResponse(content={"busy": False, "message": "无活动处理任务或 RAG 服务空闲。"}, status_code=200)
            raise HTTPException(status_code=e.response.status_code, detail=f"RAG 服务错误 (流水线状态): {e.response.text}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)