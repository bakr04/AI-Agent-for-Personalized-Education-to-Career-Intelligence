import json
import os
from functools import lru_cache
from typing import Any, Dict

try:
    import orjson  # type: ignore
except Exception:
    orjson = None
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableLambda
from packages.prompts import (
    labor_extraction_prompt,
    labor_analysis_prompt,
    labor_plan_prompt,
    labor_inference_prompt,
    labor_inference_prompt_fast,
)
from packages.models import (
    LaborData,
    LaborAnalysisOutput,
    MissingInfo,
    MarketPlan,
    InferenceOutput,
)


class LaborIntelligenceAgent:
    def __init__(self):
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_retries=2,
            timeout=45,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self._cache: Dict[str, InferenceOutput] = {}
        fast_mode = os.getenv("FAST_MODE", "").strip() == "1"
        inference_prompt = labor_inference_prompt_fast if fast_mode else labor_inference_prompt
        self.inference_chain = RunnableSequence(inference_prompt, self.llm)
        self.extraction_chain = RunnableSequence(labor_extraction_prompt, self.llm)
        self.analysis_chain = RunnableSequence(labor_analysis_prompt, self.llm)
        self.plan_chain = RunnableSequence(labor_plan_prompt, self.llm)

        self.inference_pipeline = RunnableSequence(
            RunnableLambda(self.clean_input),
            RunnableLambda(lambda cleaned: {"raw_input": cleaned}),
            self.inference_chain,
            RunnableLambda(lambda response: self._safe_json_loads(response.content)),
            RunnableLambda(self._finalize_inference_payload),
        )

    @lru_cache(maxsize=512)
    def clean_input(self, raw_input: str) -> str:
        return " ".join(raw_input.strip().split())

    @lru_cache(maxsize=512)
    def _strip_json_fence(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        return cleaned

    def _fast_json_loads(self, text: str) -> dict:
        if orjson is not None:
            return orjson.loads(text)
        return json.loads(text)

    def _safe_json_loads(self, text: str) -> dict:
        json_text = self._strip_json_fence(text)
        json_text = json_text.replace("\r", "").strip()

        def _extract_json(candidate: str) -> str:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                return candidate[start : end + 1]
            return candidate

        json_text = _extract_json(json_text)

        try:
            return self._fast_json_loads(json_text)
        except Exception:
            cleaned = "".join(ch for ch in json_text if ord(ch) >= 32 or ch in "\n\t")
            try:
                return self._fast_json_loads(cleaned)
            except Exception as exc:
                raise ValueError(f"JSON غير صالح: {exc}")

    @lru_cache(maxsize=512)
    def _join(self, items: tuple) -> str:
        return ", ".join([item for item in items if item])

    def _normalize_list(self, items) -> tuple:
        if not items:
            return ()
        return tuple(item for item in items if item)

    def _format_postings(self, labor_data: LaborData) -> str:
        if not labor_data.job_postings:
            return ""
        lines = []
        for post in labor_data.job_postings:
            pieces = []
            if post.title:
                pieces.append(f"المسمى: {post.title}")
            if post.company:
                pieces.append(f"الشركة: {post.company}")
            if post.location:
                pieces.append(f"المكان: {post.location}")
            if post.employment_type:
                pieces.append(f"نوع العمل: {post.employment_type}")
            if post.seniority:
                pieces.append(f"المستوى: {post.seniority}")
            if post.date:
                pieces.append(f"التاريخ: {post.date}")
            if post.description:
                pieces.append(f"الوصف: {post.description}")
            if post.skills:
                pieces.append(f"المهارات: {self._join(self._normalize_list(post.skills))}")
            lines.append(" | ".join(pieces))
        return "\n".join(lines)

    def _build_prompt_context(self, labor_data: LaborData) -> Dict[str, str]:
        return {
            "job_postings_summary": self._format_postings(labor_data),
        }

    def _try_parse_json_input(self, raw_input: str) -> Dict[str, Any] | None:
        raw = raw_input.strip()
        if not raw:
            return None
        if raw.startswith("{") or raw.startswith("["):
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None

    def parse_input(self, raw_input: str) -> LaborData:
        response = self.extraction_chain.invoke({"raw_input": raw_input})
        json_text = self._strip_json_fence(response.content)
        try:
            return LaborData.model_validate_json(json_text)
        except Exception as exc:
            raise ValueError(f"JSON استخراج غير صالح: {exc}")

    def analyze(self, labor_data: LaborData) -> LaborAnalysisOutput:
        response = self.analysis_chain.invoke(self._build_prompt_context(labor_data))

        json_text = self._strip_json_fence(response.content)
        try:
            return LaborAnalysisOutput.model_validate_json(json_text)
        except Exception as exc:
            raise ValueError(f"JSON التحليل غير صالح: {exc}")

    def analyze_raw_input(self, raw_input: str) -> LaborAnalysisOutput:
        labor_data = self.parse_input(raw_input)
        return self.analyze(labor_data)

    def detect_missing_info(self, labor_data: LaborData) -> MissingInfo:
        missing_fields = []
        questions = []

        if not labor_data.job_postings:
            missing_fields.append("قائمة الوظائف")
            questions.append("ممكن تبعت قائمة وظائف أو أوصاف مختصرة لكل وظيفة؟")
            return MissingInfo(missing_fields=missing_fields, questions=questions)

        def check_field(label: str, accessor) -> None:
            if not accessor:
                missing_fields.append(label)

        for post in labor_data.job_postings:
            check_field("المسمى الوظيفي", post.title)
            check_field("وصف الوظيفة", post.description)

        if "المسمى الوظيفي" in missing_fields:
            questions.append("ممكن توضح المسميات الوظيفية لكل إعلان؟")
        if "وصف الوظيفة" in missing_fields:
            questions.append("ممكن تضيف وصف مختصر للمهام أو المهارات المطلوبة؟")

        return MissingInfo(missing_fields=missing_fields, questions=questions)

    def generate_learning_plan(
        self,
        labor_data: LaborData,
        analysis: LaborAnalysisOutput,
    ) -> MarketPlan:
        context = self._build_prompt_context(labor_data)
        context["market_summary"] = analysis.market_summary
        response = self.plan_chain.invoke(context)

        json_text = self._strip_json_fence(response.content)
        try:
            return MarketPlan.model_validate_json(json_text)
        except Exception as exc:
            raise ValueError(f"JSON خطة السوق غير صالح: {exc}")

    def _finalize_inference_payload(self, inference_payload: dict) -> dict:
        extracted = LaborData.model_validate(inference_payload.get("extracted", {}))
        missing_info = inference_payload.get("missing_info")
        should_recompute = False
        if not missing_info:
            should_recompute = True
        elif isinstance(missing_info, dict):
            missing_fields = missing_info.get("missing_fields", [])
            questions = missing_info.get("questions", [])
            if not missing_fields and not questions:
                should_recompute = True

        if should_recompute:
            inference_payload["missing_info"] = self.detect_missing_info(extracted).model_dump()
        return inference_payload

    def _is_sparse_extraction(self, extracted: LaborData) -> bool:
        if not extracted.job_postings:
            return True
        filled = 0
        for post in extracted.job_postings:
            if post.title:
                filled += 1
            if post.description:
                filled += 1
            if post.skills:
                filled += 1
        return filled < 2

    def _fallback_full_pipeline(self, raw_input: str) -> InferenceOutput:
        labor_data = self.parse_input(raw_input)
        analysis = self.analyze(labor_data)
        plan = self.generate_learning_plan(labor_data, analysis)
        missing_info = self.detect_missing_info(labor_data)
        return InferenceOutput(
            extracted=labor_data,
            analysis=analysis,
            learning_plan=plan,
            missing_info=missing_info,
        )

    def infer(self, raw_input: str) -> InferenceOutput:
        cleaned_input = self.clean_input(raw_input)
        if cleaned_input in self._cache:
            return self._cache[cleaned_input]

        direct_payload = self._try_parse_json_input(cleaned_input)
        if direct_payload:
            try:
                output = InferenceOutput.model_validate(direct_payload)
                self._cache[cleaned_input] = output
                return output
            except Exception:
                try:
                    labor_data = LaborData.model_validate(direct_payload)
                    analysis = self.analyze(labor_data)
                    plan = self.generate_learning_plan(labor_data, analysis)
                    missing_info = self.detect_missing_info(labor_data)
                    output = InferenceOutput(
                        extracted=labor_data,
                        analysis=analysis,
                        learning_plan=plan,
                        missing_info=missing_info,
                    )
                    self._cache[cleaned_input] = output
                    return output
                except Exception:
                    pass

        inference_payload = self.inference_pipeline.invoke(raw_input)
        output = InferenceOutput.model_validate(inference_payload)

        if self._is_sparse_extraction(output.extracted):
            output = self._fallback_full_pipeline(raw_input)

        self._cache[cleaned_input] = output
        return output

    def infer_pipeline(self, raw_input: str) -> InferenceOutput:
        return self.infer(raw_input)
