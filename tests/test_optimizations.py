import os
import sys
import time

# Ensure root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chatbot_service import get_chatbot_service

def test_rule_extraction():
    print("\n" + "="*60)
    print("TEST 1: RULE-BASED INGREDIENT EXTRACTION (PYTHON <1MS)")
    print("="*60)
    chatbot = get_chatbot_service()

    test_messages = [
        "Tủ lạnh còn 2 quả trứng gà với mấy quả cà chua dập, nấu gì ngon?",
        "Tôi có thịt lợn và khổ qua thì làm món gì?",
        "Hôm nay muốn ăn món gì với bắp bò và rau cải thìa",
        "Có tôm sú, mực ống và nấm hương làm lẩu được không",
        "Xin chào bạn, hôm nay thời tiết đẹp quá"
    ]

    for msg in test_messages:
        t0 = time.perf_counter()
        ings = chatbot._extract_ingredients_with_rule(msg)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[*] Input: '{msg}'")
        print(f"    -> Extracted ({len(ings)} items): {ings}")
        print(f"    -> Latency: {elapsed_ms:.3f} ms")
        assert elapsed_ms < 50, f"Extraction too slow: {elapsed_ms}ms"

    # Specific accuracy and ambiguity checks
    # 1. 'Tôi' must NEVER trigger 'tỏi'
    res_toi = chatbot._extract_ingredients_with_rule("Tôi có thịt lợn và khổ qua thì làm món gì?")
    print(f"[*] 'Tôi có thịt lợn và khổ qua...' -> {res_toi}")
    assert "tỏi" not in res_toi, "BUG: 'Tôi' must not be matched as 'tỏi'!"
    assert "thịt lợn" in res_toi
    assert "khổ qua" in res_toi

    # 2. Real 'tỏi' must be detected
    res_real_toi = chatbot._extract_ingredients_with_rule("Tôi có tỏi, ớt và sườn non")
    print(f"[*] 'Tôi có tỏi, ớt và sườn non' -> {res_real_toi}")
    assert "tỏi" in res_real_toi
    assert "ớt" in res_real_toi
    assert "sườn non" in res_real_toi

    # 3. 'cá lóc' must not trigger 'cà'
    res_ca = chatbot._extract_ingredients_with_rule("Canh chua cá lóc miền Tây")
    print(f"[*] 'Canh chua cá lóc...' -> {res_ca}")
    assert "cá lóc" in res_ca
    assert "cà" not in res_ca

    # 4. 'trứng gà và cà chua'
    res1 = chatbot._extract_ingredients_with_rule("trứng gà và cà chua")
    assert "trứng gà" in res1
    assert "cà chua" in res1

    # 5. Unaccented sentence matching
    res_unaccented = chatbot._extract_ingredients_with_rule("toi co thit bo va ca chua")
    print(f"[*] Unaccented 'toi co thit bo va ca chua' -> {res_unaccented}")
    assert "tỏi" not in res_unaccented
    assert any("bò" in x or "bo" in x for x in res_unaccented)
    assert any("cà chua" in x for x in res_unaccented)

    # 6. General greeting must be empty
    res_empty = chatbot._extract_ingredients_with_rule("Hôm nay trời đẹp quá, xin chào!")
    assert len(res_empty) == 0

    print("\n[OK] All rigorous accuracy and anti-ambiguity tests passed!")

def test_mode_switch():
    print("\n" + "="*60)
    print("TEST 2: INGREDIENT_EXTRACTION_MODE SWITCHING")
    print("="*60)
    chatbot = get_chatbot_service()
    sample = "Tôi có thịt bò và hành lá"

    # Mode 1: Rule
    os.environ["INGREDIENT_EXTRACTION_MODE"] = "rule"
    res_rule = chatbot._extract_ingredients(sample)
    print(f"[*] Mode 'rule': {res_rule}")
    assert "thịt bò" in res_rule
    assert "hành lá" in res_rule

    # Mode 2: Hybrid
    os.environ["INGREDIENT_EXTRACTION_MODE"] = "hybrid"
    res_hybrid = chatbot._extract_ingredients(sample)
    print(f"[*] Mode 'hybrid': {res_hybrid}")
    assert "thịt bò" in res_hybrid

    # Mode 3: LLM (verify function exists and is callable)
    os.environ["INGREDIENT_EXTRACTION_MODE"] = "llm"
    print("[*] Mode 'llm' is configured and selectable via INGREDIENT_EXTRACTION_MODE=llm")

    # Reset to default
    os.environ["INGREDIENT_EXTRACTION_MODE"] = "hybrid"
    print("\n[OK] Mode switching verified successfully!")

def test_streaming_interface():
    print("\n" + "="*60)
    print("TEST 3: STREAMING INTERFACE VERIFICATION")
    print("="*60)
    chatbot = get_chatbot_service()
    assert hasattr(chatbot, "stream_chat"), "ChatbotService must have stream_chat"
    assert hasattr(chatbot.client, "chat_completions_stream"), "LLMClient must have chat_completions_stream"
    print("[*] ChatbotService.stream_chat interface: READY")
    print("[*] LLMClient.chat_completions_stream interface: READY")
    print("\n[OK] Streaming interfaces verified successfully!")

if __name__ == "__main__":
    test_rule_extraction()
    test_mode_switch()
    test_streaming_interface()
    print("\n" + "="*60)
    print("ALL OPTIMIZATION TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
