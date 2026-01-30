import json
import sys

try:
    import orjson  # type: ignore
except Exception:
    orjson = None
from dotenv import load_dotenv
from packages.agent import LaborIntelligenceAgent

RAW_INPUT = """
ده ملخص لعدد من إعلانات الوظائف (نص خام)، وعايز منك تحلل سوق العمل بناءً عليهم.

إعلان 1:
- المسمى: عالم بيانات
- التاريخ: 2026-01-15
- الوصف: خبرة في بايثون وSQL وتعلم الآلة والإحصاء

إعلان 2:
- المسمى: مهندس ذكاء اصطناعي
- التاريخ: 2026-01-20
- الوصف: مستوى قوي في بايثون، تعلم عميق، معالجة لغة طبيعية، وخبرة كلاود

إعلان 3:
- المسمى: محلل بيانات
- التاريخ: 2026-01-10
- الوصف: إكسل وSQL وPower BI وتصوير البيانات
""".strip()

load_dotenv()


def _print_json(payload: dict) -> None:
    if orjson is not None:
        print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    agent = LaborIntelligenceAgent()
    try:
        output = agent.infer(RAW_INPUT)
    except Exception as exc:
        # لا نطبع أي شيء غير JSON حسب المطلوب
        sys.exit(1)

    _print_json(output.model_dump())


if __name__ == "__main__":
    main()
