"""REST API 层：把检测能力以接口形式暴露，便于前后端分离与联调。

启动：uvicorn api:app --reload
文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from data_gen import load_readings, load_truth
from detect import evaluate_all, run_pipeline

STATE: dict = {}


def build_state() -> dict:
    readings = load_readings()
    anomalies = run_pipeline(readings)
    truth = load_truth()
    return {
        "readings": readings,
        "anomalies": anomalies,
        "metrics": evaluate_all(readings, truth) if truth is not None else None,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE.update(build_state())
    yield


app = FastAPI(
    title="工业传感器异常检测 API",
    description="模拟设备时序数据的异常检测服务，提供指标、告警区间与原始序列查询",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "devices": sorted(STATE["readings"]["device_id"].unique().tolist()),
        "rows": int(len(STATE["readings"])),
        "anomaly_segments": int(len(STATE["anomalies"])),
    }


@app.get("/api/metrics")
def metrics() -> dict:
    """全局评估指标；未生成 ground_truth.csv 时返回提示。"""
    if STATE.get("metrics") is None:
        return {"available": False,
                "hint": "先运行 python data_gen.py 生成标签文件"}
    return {"available": True, **STATE["metrics"]}


@app.get("/api/anomalies")
def anomalies(device_id: str | None = Query(default=None),
              limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    df = STATE["anomalies"]
    if device_id:
        df = df[df["device_id"] == device_id]
    return df.head(limit).to_dict("records")


@app.get("/api/series")
def series(device_id: str = Query(...),
           limit: int = Query(default=2000, ge=1, le=20000)) -> list[dict]:
    df = STATE["readings"]
    df = df[df["device_id"] == device_id].head(limit).copy()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.to_dict("records")
