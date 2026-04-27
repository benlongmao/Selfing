from fastapi import APIRouter, Query
from typing import Optional, Dict
from backend.meta_rule_learner import MetaRuleLearner
from backend.database import DB_PATH

router = APIRouter(tags=["MetaRules"])

@router.get("/meta-rules")
def get_meta_rules(
    category: Optional[str] = Query(None, description="Filter: learning | modification | compression | selection"),
    limit: int = Query(20, ge=1, le=100)
):
    learner = MetaRuleLearner(DB_PATH)
    meta_rules = learner.get_meta_rules(category=category, limit=limit)
    return {"meta_rules": meta_rules, "count": len(meta_rules)}

@router.post("/meta-rules/learn")
def learn_meta_rule(body: Dict):
    learner = MetaRuleLearner(DB_PATH)
    meta_rule = learner.learn_from_experience(body)
    if meta_rule:
        return {
            "ok": True,
            "meta_rule": {
                "id": meta_rule.id,
                "text": meta_rule.text,
                "category": getattr(meta_rule, "category", "learning")
            }
        }
    else:
        return {"ok": False, "message": "No meta-rule generated from this experience"}

@router.post("/meta-rules/apply")
def apply_meta_rules(body: Dict):
    learner = MetaRuleLearner(DB_PATH)
    return learner.apply_meta_rules(
        action_type=body.get("action_type"),
        context=body.get("context", {})
    )

@router.post("/meta-rules/cleanup")
def cleanup_meta_rules(body: Optional[Dict] = None):
    learner = MetaRuleLearner(DB_PATH)
    min_success_rate = 0.2
    min_evidence = 5
    if body:
        min_success_rate = body.get("min_success_rate", 0.2)
        min_evidence = body.get("min_evidence", 5)
    return learner.evaluate_and_cleanup_meta_rules(
        min_success_rate=min_success_rate,
        min_evidence=min_evidence
    )

