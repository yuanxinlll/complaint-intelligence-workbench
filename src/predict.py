"""Predict a single complaint locally, without uploading its text."""

import argparse
import hashlib
import json

import joblib

from src.data import ROOT, clean_text
from src.review import review_flags


def predict_text(text, model, review_rule=None):
    if not clean_text(text):
        raise ValueError("请输入非空投诉正文。")
    if len(text) > 50000:
        raise ValueError("正文超过 50,000 字符限制，请缩短后重试。")
    scores = model.predict_proba([text])[0]
    index = int(scores.argmax())
    flag = bool(review_flags([scores[index]], review_rule)[0]) if review_rule else None
    status = "按验证集规则产生提示；未标记不代表可以放心自动处理。" if review_rule else "尚未加载匹配模型的复核规则，暂不生成复核标记。"
    return {"predicted_issue": str(model.classes_[index]), "model_score": round(float(scores[index]), 6), "needs_review": flag, "review_status": status, "score_note": "模型分数不是正确率，也不是经过校准的真实正确概率。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    model_path = ROOT / "artifacts/model.joblib"
    try:
        if not model_path.exists():
            raise ValueError("模型不存在，请先完成阶段 2 训练。")
        rule_path = ROOT / "artifacts/review_rule.json"
        rule = json.loads(rule_path.read_text()) if rule_path.exists() else None
        if rule and rule.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError("复核规则与当前模型不匹配，请重新验证后使用。")
        result = predict_text(args.text, joblib.load(model_path), rule)
    except ValueError as error:
        parser.exit(2, f"错误：{error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
