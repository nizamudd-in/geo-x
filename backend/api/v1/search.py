"""
GeoSense-X — Interactive Search Endpoints
==========================================
WebSocket endpoint for progressive result delivery and interactive refinement.

  WS  /search/interactive → streaming phased search results
  POST /search/refine     → axis-weighted query refinement
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from core.logging import get_logger
from models.schemas.retrieval import RefineRequest, RetrievalResponse

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/interactive")
async def interactive_search(
    websocket: WebSocket,
    request: Request,
) -> None:
    """
    WebSocket endpoint for interactive search with progressive result delivery.

    Message protocol:
      Client → Server:
        { "type": "QUERY", "image_b64": "...", "source_modality": "optical",
          "target_modality": "sar", "top_k": 10, "explanation_depth": "full" }
        { "type": "REFINE", "query_id": "...", "axis": "urban", "direction": "more" }
        { "type": "EXPLORE_HOP", "query_id": "...", "pivot_scene_id": "..." }

      Server → Client (phased):
        Phase 1: { "type": "EMBEDDING_READY", "query_id": "..." }          ~50ms
        Phase 2: { "type": "TOP_K_RESULTS", "results": [...] }            ~150ms
        Phase 3: { "type": "AXES_READY", "results": [...] }               ~200ms
        Phase 4: { "type": "HEATMAP_READY", "scene_id": "...", "urls": {}  ~500ms/result
        Error:   { "type": "ERROR", "code": "...", "detail": "..." }

    Connection timeout: 30 seconds of inactivity.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        # TODO: implement authentication from query param ?token=...
        # token = websocket.query_params.get("token")
        # user = await validate_ws_token(token)

        while True:
            raw_message = await websocket.receive_text()

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "ERROR",
                    "code": "INVALID_JSON",
                    "detail": "Message must be valid JSON",
                })
                continue

            msg_type = message.get("type")

            if msg_type == "QUERY":
                await _handle_query_message(websocket, message, request)
            elif msg_type == "REFINE":
                await _handle_refine_message(websocket, message, request)
            elif msg_type == "EXPLORE_HOP":
                await _handle_hop_message(websocket, message, request)
            else:
                await websocket.send_json({
                    "type": "ERROR",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "detail": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error", error=str(exc))
        try:
            await websocket.send_json({
                "type": "ERROR",
                "code": "SERVER_ERROR",
                "detail": "An unexpected error occurred",
            })
        except Exception:
            pass


async def _handle_query_message(
    websocket: WebSocket, message: dict, request: Request
) -> None:
    """
    Handle a QUERY message from the WebSocket client.

    Runs the retrieval pipeline and sends phased results.
    """
    query_id = str(uuid.uuid4())

    # Phase 1: acknowledge query received
    await websocket.send_json({"type": "EMBEDDING_READY", "query_id": query_id})

    # TODO: Phase 2: run FAISS search, stream top-K results
    # await websocket.send_json({"type": "TOP_K_RESULTS", "results": [...]})

    # TODO: Phase 3: run axis scoring, stream explanations
    # await websocket.send_json({"type": "AXES_READY", "results": [...]})

    # TODO: Phase 4: stream heatmaps as they complete
    # for scene_id in result_scene_ids:
    #     await websocket.send_json({
    #         "type": "HEATMAP_READY",
    #         "scene_id": scene_id,
    #         "heatmap_urls": {...},
    #     })


async def _handle_refine_message(
    websocket: WebSocket, message: dict, request: Request
) -> None:
    """Handle axis refinement request from WebSocket client."""
    # TODO: implement axis-weighted embedding refinement
    await websocket.send_json({
        "type": "ERROR",
        "code": "NOT_IMPLEMENTED",
        "detail": "REFINE not yet implemented",
    })


async def _handle_hop_message(
    websocket: WebSocket, message: dict, request: Request
) -> None:
    """Handle multi-hop exploration from WebSocket client."""
    # TODO: implement result-as-new-query hop
    await websocket.send_json({
        "type": "ERROR",
        "code": "NOT_IMPLEMENTED",
        "detail": "EXPLORE_HOP not yet implemented",
    })


@router.post(
    "/refine",
    response_model=RetrievalResponse,
    summary="Refine search by semantic axis weights",
    description="Adjust the query embedding by semantic axis direction vectors "
                "and return new top-K results. Use to 'search for more urban' "
                "or 'search for less vegetation'.",
)
async def refine_search(
    request: Request,
    body: RefineRequest,
) -> RetrievalResponse:
    """
    Apply axis-weighted refinement to an existing query embedding.

    Loads the original query_embedding from cache/DB, applies axis direction
    vector offsets with the specified weights and directions, then re-runs
    FAISS search with the modified embedding.
    """
    # TODO: implement axis refinement
    # z_refined = original_embedding.copy()
    # for ref in body.refinements:
    #     direction_sign = 1.0 if ref.direction == "more" else -1.0
    #     axis_vec = AxisScorer.get_axis_vector(ref.axis)
    #     z_refined += direction_sign * ref.weight * axis_vec
    # z_refined = z_refined / np.linalg.norm(z_refined)
    # return await faiss_service.search(z_refined, top_k=body.top_k)
    raise NotImplementedError("Search refinement not yet implemented")
