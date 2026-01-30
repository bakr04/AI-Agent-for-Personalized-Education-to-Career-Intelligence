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
    learner_analysis_prompt,
    learner_extraction_prompt,
    learner_plan_prompt,
    learner_inference_prompt,
    learner_inference_prompt_fast,
)
from packages.models import (
    LearnerData,
    LearnerAnalysisOutput,
    MissingInfo,
    LearningPlan,
    InferenceOutput,
)

class LearnerIntelligenceAgent:
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
        inference_prompt = learner_inference_prompt_fast if fast_mode else learner_inference_prompt
        self.inference_chain = RunnableSequence(inference_prompt, self.llm)
        self.extraction_chain = RunnableSequence(learner_extraction_prompt, self.llm)
        self.analysis_chain = RunnableSequence(learner_analysis_prompt, self.llm)
        self.plan_chain = RunnableSequence(learner_plan_prompt, self.llm)

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

    def _build_prompt_context(self, learner_data: LearnerData) -> Dict[str, str]:
        return {
            "courses": self._join(self._normalize_list(learner_data.courses)),
            "grades": self._join(self._normalize_list(learner_data.grades)),
            "skills": self._join(self._normalize_list(learner_data.skills)),
            "interests": self._join(self._normalize_list(learner_data.interests)),
            "research_goals": self._join(self._normalize_list(learner_data.research_goals)),
            "education_level": learner_data.education_level or "",
            "certifications": self._join(self._normalize_list(learner_data.certifications)),
            "projects": self._join(self._normalize_list(learner_data.projects)),
            "tools": self._join(self._normalize_list(learner_data.tools)),
            "languages": self._join(self._normalize_list(learner_data.languages)),
            "preferred_domains": self._join(self._normalize_list(learner_data.preferred_domains)),
            "availability": learner_data.availability or "",
            "learning_preferences": self._join(self._normalize_list(learner_data.learning_preferences)),
            "constraints": self._join(self._normalize_list(learner_data.constraints)),
        }

    def _try_parse_json_input(self, raw_input: str) -> Dict[str, Any] | None:
        raw = raw_input.strip()
        if not raw.startswith("{"):
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def parse_input(self, raw_input: str) -> LearnerData:
        response = self.extraction_chain.invoke({"raw_input": raw_input})
        json_text = self._strip_json_fence(response.content)
        try:
            return LearnerData.model_validate_json(json_text)
        except Exception as exc:
            raise ValueError(f"JSON استخراج غير صالح: {exc}")

    def analyze(self, learner_data: LearnerData) -> LearnerAnalysisOutput:
        response = self.analysis_chain.invoke(self._build_prompt_context(learner_data))

        json_text = self._strip_json_fence(response.content)
        try:
            return LearnerAnalysisOutput.model_validate_json(json_text)
        except Exception as exc:
            raise ValueError(f"JSON التحليل غير صالح: {exc}")

    def analyze_raw_input(self, raw_input: str) -> LearnerAnalysisOutput:
        learner_data = self.parse_input(raw_input)
        return self.analyze(learner_data)

    def detect_missing_info(self, learner_data: LearnerData) -> MissingInfo:
        missing_fields = []
        questions = []

        def check_list(field_name: str, label: str, example: str) -> None:
            if not getattr(learner_data, field_name):
                missing_fields.append(label)
                questions.append(f"ممكن تقولّي {label} عندك؟ (مثال: {example})")

        def check_scalar(field_name: str, label: str, example: str) -> None:
            if not getattr(learner_data, field_name):
                missing_fields.append(label)
                questions.append(f"ممكن تقولّي {label}؟ (مثال: {example})")

        check_list("courses", "الكورسات اللي درستها", "تعلم الآلة، قواعد البيانات")
        check_list("grades", "الدرجات في الكورسات", "تعلم الآلة: A، نظم التشغيل: B+")
        check_list("skills", "المهارات", "بايثون، SQL، لينكس")
        check_list("interests", "اهتماماتك", "بحث في الذكاء الاصطناعي، MLOps")
        check_list("research_goals", "أهدافك البحثية", "نشر ورقة، بناء أنظمة")
        check_scalar("education_level", "مستواك التعليمي", "طالب سنة رابعة علوم كمبيوتر")
        check_list("certifications", "الشهادات", "AWS CCP، TensorFlow")
        check_list("projects", "المشاريع اللي اشتغلت عليها", "نظام توصية")
        check_list("tools", "الأدوات اللي بتستخدمها", "Git، Docker")
        check_list("languages", "اللغات اللي بتتكلمها", "العربي، الإنجليزي")
        check_list("preferred_domains", "المجالات المفضلة", "NLP، تعلم آلي موزع")
        check_scalar("availability", "الوقت المتاح أسبوعيًا", "10 ساعات في الأسبوع")
        check_list("learning_preferences", "تفضيلات التعلم", "عملي، مشاريع")
        check_list("constraints", "القيود", "وقت محدود، مافيش GPU")

        return MissingInfo(missing_fields=missing_fields, questions=questions)

    def generate_learning_plan(
        self,
        learner_data: LearnerData,
        analysis: LearnerAnalysisOutput,
    ) -> LearningPlan:
        context = self._build_prompt_context(learner_data)
        context["summarized_report"] = analysis.summarized_report
        response = self.plan_chain.invoke(context)

        json_text = self._strip_json_fence(response.content)
        try:
            return LearningPlan.model_validate_json(json_text)
        except Exception as exc:
            raise ValueError(f"JSON خطة التعلم غير صالح: {exc}")

    def _finalize_inference_payload(self, inference_payload: dict) -> dict:
        extracted = LearnerData.model_validate(inference_payload.get("extracted", {}))
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

    def _is_sparse_extraction(self, extracted: LearnerData) -> bool:
        filled_counts = sum(
            1
            for items in (
                extracted.courses,
                extracted.grades,
                extracted.skills,
                extracted.interests,
                extracted.research_goals,
                extracted.certifications,
                extracted.projects,
                extracted.tools,
                extracted.languages,
                extracted.preferred_domains,
                extracted.learning_preferences,
                extracted.constraints,
            )
            if items
        )
        scalar_counts = sum(
            1
            for value in (
                extracted.education_level,
                extracted.availability,
            )
            if value
        )
        return (filled_counts + scalar_counts) < 3

    def _fallback_full_pipeline(self, raw_input: str) -> InferenceOutput:
        learner_data = self.parse_input(raw_input)
        analysis = self.analyze(learner_data)
        plan = self.generate_learning_plan(learner_data, analysis)
        missing_info = self.detect_missing_info(learner_data)
        return InferenceOutput(
            extracted=learner_data,
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
                pass

        inference_payload = self.inference_pipeline.invoke(raw_input)
        output = InferenceOutput.model_validate(inference_payload)

        if self._is_sparse_extraction(output.extracted):
            output = self._fallback_full_pipeline(raw_input)

        self._cache[cleaned_input] = output
        return output

    def infer_pipeline(self, raw_input: str) -> InferenceOutput:
        return self.infer(raw_input)
