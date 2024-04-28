from fastapi import FastAPI, Body, Request
from pydantic import BaseModel
import spacy
import uvicorn
import time

from prometheus_client import Summary, start_http_server, Counter, Gauge
from prometheus_client import disable_created_metrics

