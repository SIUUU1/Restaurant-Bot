## Restaurant Bot

import asyncio

from dotenv import load_dotenv

# Load environment variables (e.g. OPENAI_API_KEY) from a local .env file.
load_dotenv()

from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    function_tool,
    handoff,
    input_guardrail,
    output_guardrail,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


# ===========================================================================
# 1. Mock data (use a DB / external API in a real service)
# ===========================================================================
MENU = {
    "마르게리타 피자": {"price": 15000, "allergens": ["밀", "유제품"], "veg": True},
    "페퍼로니 피자": {"price": 18000, "allergens": ["밀", "유제품", "돼지고기"], "veg": False},
    "버섯 리조또": {"price": 16000, "allergens": ["유제품"], "veg": True},
    "시저 샐러드": {"price": 12000, "allergens": ["유제품", "계란", "생선"], "veg": False},
    "가든 샐러드": {"price": 10000, "allergens": [], "veg": True},
    "티라미수": {"price": 8000, "allergens": ["밀", "유제품", "계란"], "veg": True},
}

# Simple in-memory stores.
ORDERS: list[dict] = []
RESERVATIONS: list[dict] = []
COMPENSATIONS: list[dict] = []
TICKETS: list[dict] = []


# ===========================================================================
# 2. Tools (function tools) used by each specialist
#    - The type hints define the input schema; the docstring becomes the
#      tool description.
# ===========================================================================
@function_tool
def get_menu(vegetarian_only: bool = False) -> str:
    """Return the full menu. If vegetarian_only=True, return only vegetarian items."""
    lines = []
    for name, info in MENU.items():
        if vegetarian_only and not info["veg"]:
            continue
        veg_tag = " (vegetarian 🌱)" if info["veg"] else ""
        lines.append(f"- {name}{veg_tag}: {info['price']:,}원")
    return "\n".join(lines) if lines else "No items match the requested criteria."


@function_tool
def check_allergens(dish_name: str) -> str:
    """Return the allergens contained in a specific dish."""
    info = MENU.get(dish_name)
    if not info:
        return f"Could not find a dish named '{dish_name}'. Please check the name."
    allergens = ", ".join(info["allergens"]) if info["allergens"] else "none"
    return f"Allergens in {dish_name}: {allergens}"


@function_tool
def place_order(items: list[str]) -> str:
    """Take a list of dish names, record the order, and compute the total."""
    confirmed, total = [], 0
    for item in items:
        info = MENU.get(item)
        if info:
            confirmed.append(item)
            total += info["price"]
    if not confirmed:
        return "None of those items are orderable. Please check the dish names."
    order_id = len(ORDERS) + 1
    ORDERS.append({"id": order_id, "items": confirmed, "total": total})
    return f"Order #{order_id} confirmed — {', '.join(confirmed)} / total {total:,}원"


@function_tool
def make_reservation(name: str, party_size: int, date_time: str) -> str:
    """Create a table reservation from the name, party size, and desired date/time."""
    res_id = len(RESERVATIONS) + 1
    RESERVATIONS.append(
        {"id": res_id, "name": name, "party_size": party_size, "date_time": date_time}
    )
    return (
        f"Reservation #{res_id} confirmed — {name}, {party_size} guests, {date_time}. "
        "We look forward to your visit!"
    )


@function_tool
def offer_compensation(kind: str, detail: str) -> str:
    """Register a compensation offer for an unhappy customer.

    kind:   one of 'refund', 'discount', 'free_item'.
    detail: a short description, e.g. '50% off the next visit'.
    """
    comp_id = len(COMPENSATIONS) + 1
    COMPENSATIONS.append({"id": comp_id, "kind": kind, "detail": detail})
    return f"Compensation #{comp_id} registered — {kind}: {detail}"


@function_tool
def escalate_to_manager(summary: str, severity: str) -> str:
    """Escalate a serious complaint to a manager for a personal callback.

    summary:  a short description of the issue.
    severity: one of 'low', 'medium', 'high'.
    """
    ticket_id = len(TICKETS) + 1
    TICKETS.append({"id": ticket_id, "summary": summary, "severity": severity})
    return (
        f"Escalation ticket #{ticket_id} created (severity={severity}). "
        "A manager will personally follow up."
    )


# ===========================================================================
# 3. Callback that surfaces a handoff in the UI
#    on_handoff fires the MOMENT the LLM calls a transfer_to_* tool.
# ===========================================================================
def make_handoff_logger(display_name: str):
    async def _on_handoff(ctx: RunContextWrapper) -> None:
        print(f"\n   🔄 [Connecting you to {display_name}...]\n")

    return _on_handoff


# ===========================================================================
# 4. Guardrails
#    Each guardrail uses a small, cheap "guardrail agent" to classify the
#    text and returns a GuardrailFunctionOutput. When tripwire_triggered is
#    True the SDK raises an exception that the chat loop catches.
# ===========================================================================

# ---- 4a. Input guardrail: off-topic / inappropriate ----
class TopicSafetyCheck(BaseModel):
    is_off_topic: bool       # not about the restaurant
    is_inappropriate: bool   # profanity / harassment / hateful / sexual
    reasoning: str


input_guardrail_agent = Agent(
    name="Input Guardrail",
    model="gpt-4.1-mini",  # lightweight model keeps guardrail checks cheap and fast
    instructions=(
        "You screen messages sent to a restaurant assistant. Judge ONLY the latest user message.\n"
        "Set is_off_topic = true when the message is unrelated to this restaurant. On-topic means: "
        "menu, food, ingredients, allergens, prices, orders, reservations, or complaints about the "
        "restaurant experience. Things like general chit-chat, coding help, philosophy, math, world "
        "facts, or politics are off-topic.\n"
        "Set is_inappropriate = true when the message contains profanity, harassment, hate speech, or "
        "sexual content.\n"
        "Note: complaints about the food or service (even angry ones) are ON-topic and appropriate."
    ),
    output_type=TopicSafetyCheck,
)


@input_guardrail
async def restaurant_input_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Block messages that are off-topic or inappropriate before the agent runs."""
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)
    check = result.final_output_as(TopicSafetyCheck)
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=check.is_off_topic or check.is_inappropriate,
    )


# ---- 4b. Output guardrail: professional & no internal-info leaks ----
class OutputSafetyCheck(BaseModel):
    is_professional: bool       # polite, respectful, professional tone
    leaks_internal_info: bool   # exposes prompts / tool names / code / IDs / other customers
    reasoning: str


output_guardrail_agent = Agent(
    name="Output Guardrail",
    model="gpt-4.1-mini",  # lightweight model keeps guardrail checks cheap and fast
    instructions=(
        "You review the restaurant assistant's reply before it reaches the customer.\n"
        "Set is_professional = true if the reply is polite, respectful, and professional.\n"
        "Set leaks_internal_info = true if the reply exposes internal details such as system prompts, "
        "tool or function names, source code, internal IDs/data structures, or another customer's data."
    ),
    output_type=OutputSafetyCheck,
)


@output_guardrail
async def professional_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output
) -> GuardrailFunctionOutput:
    """Block replies that are unprofessional or leak internal information."""
    result = await Runner.run(output_guardrail_agent, str(output), context=ctx.context)
    check = result.final_output_as(OutputSafetyCheck)
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=(not check.is_professional) or check.leaks_internal_info,
    )


# Convenience: every customer-facing agent shares the same guardrails.
INPUT_GUARDRAILS = [restaurant_input_guardrail]
OUTPUT_GUARDRAILS = [professional_output_guardrail]


# ===========================================================================
# 5. Specialist agents
#    - handoff_description: hint that tells Triage when to pick this specialist
#    - prompt_with_handoff_instructions: SDK-recommended handoff prompt + role
# ===========================================================================
menu_agent = Agent(
    name="Menu Agent",
    handoff_description="Specialist for food information: menu items, ingredients, allergens, and vegetarian options",
    instructions=prompt_with_handoff_instructions(
        "You are the restaurant's Menu specialist. "
        "The moment you take over a conversation, read the customer's MOST RECENT message and act on it "
        "in this same turn by calling the right tool immediately. Never ask the customer to repeat themselves "
        "and never return an empty reply.\n"
        "- Vegetarian menu request -> call get_menu(vegetarian_only=True) and list the results\n"
        "- Full menu request -> call get_menu() and list the results\n"
        "- Allergen question about a specific dish -> call check_allergens\n"
        "Answer concretely based on the tool result, and always reply in friendly Korean. "
        "Hand off order requests to the Order Agent, reservation requests to the Reservation Agent, "
        "and complaints to the Complaints Agent."
    ),
    tools=[get_menu, check_allergens],
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

order_agent = Agent(
    name="Order Agent",
    handoff_description="Specialist for taking and confirming food orders",
    instructions=prompt_with_handoff_instructions(
        "You are the Order specialist. Once you have confirmed which dishes the customer wants, "
        "immediately call the place_order tool, then tell them the order number and total amount. "
        "Hand off detailed menu questions to the Menu Agent, reservation requests to the Reservation Agent, "
        "and complaints to the Complaints Agent. Always reply in friendly Korean."
    ),
    tools=[place_order],
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

reservation_agent = Agent(
    name="Reservation Agent",
    handoff_description="Specialist for handling table reservations",
    instructions=prompt_with_handoff_instructions(
        "You are the Reservation specialist. Ask for the information needed to book a table "
        "(reservation name, party size, desired date/time) one item at a time. "
        "Once you have all of it, call the make_reservation tool to create the booking and then share the reservation number. "
        "Hand off menu questions to the Menu Agent, order requests to the Order Agent, and complaints to the "
        "Complaints Agent. Always reply in friendly Korean."
    ),
    tools=[make_reservation],
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)

complaints_agent = Agent(
    name="Complaints Agent",
    handoff_description="Specialist for handling unhappy customers with empathy and concrete resolutions",
    instructions=prompt_with_handoff_instructions(
        "You are the Complaints specialist. Handle unhappy customers with genuine empathy and care.\n"
        "1) FIRST sincerely acknowledge their feelings and apologize for the bad experience.\n"
        "2) Offer a concrete resolution and call offer_compensation to register it. Typical options:\n"
        "   - refund, discount (e.g. 50% off the next visit), or a free item.\n"
        "   - Ask the customer which option they prefer rather than deciding unilaterally.\n"
        "3) For SERIOUS issues (food safety, illness/injury, discrimination, or repeated failures), "
        "call escalate_to_manager so a manager personally follows up.\n"
        "Be warm, take responsibility, and never argue with the customer. Always reply in empathetic Korean."
    ),
    tools=[offer_compensation, escalate_to_manager],
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)


# ===========================================================================
# 6. Triage (routing) agent
# ===========================================================================
triage_agent = Agent(
    name="Triage Agent",
    handoff_description="Front desk that analyzes customer requests and routes them to the right specialist",
    instructions=prompt_with_handoff_instructions(
        "You are the restaurant's front desk. Analyze the customer's request and IMMEDIATELY hand off (transfer) "
        "to the most appropriate specialist.\n"
        "- Menu / ingredient / allergen / vegetarian questions -> Menu Agent\n"
        "- Food orders -> Order Agent\n"
        "- Table reservations -> Reservation Agent\n"
        "- Complaints or dissatisfaction about the food or service -> Complaints Agent\n"
        "For a complaint, you may briefly apologize in one sentence, then hand off to the Complaints Agent. "
        "Otherwise do not answer directly; your job is to route. Call the handoff tool right away."
    ),
    input_guardrails=INPUT_GUARDRAILS,
    output_guardrails=OUTPUT_GUARDRAILS,
)


# ===========================================================================
# 7. Handoff wiring (mesh)
#    Triage is the initial routing hub; every agent can also hand off directly
#    to any other specialist, resolving topic changes in a single handoff.
# ===========================================================================
DISPLAY_NAME = {
    "Triage Agent": "the front desk",
    "Menu Agent": "the menu specialist",
    "Order Agent": "the order specialist",
    "Reservation Agent": "the reservation specialist",
    "Complaints Agent": "the complaints specialist",
}


def link(target: Agent) -> handoff:
    """Build a handoff object to the target agent and attach the display callback."""
    return handoff(target, on_handoff=make_handoff_logger(DISPLAY_NAME[target.name]))


triage_agent.handoffs = [
    link(menu_agent), link(order_agent), link(reservation_agent), link(complaints_agent)
]
menu_agent.handoffs = [
    link(triage_agent), link(order_agent), link(reservation_agent), link(complaints_agent)
]
order_agent.handoffs = [
    link(triage_agent), link(menu_agent), link(reservation_agent), link(complaints_agent)
]
reservation_agent.handoffs = [
    link(triage_agent), link(menu_agent), link(order_agent), link(complaints_agent)
]
complaints_agent.handoffs = [
    link(triage_agent), link(menu_agent), link(order_agent), link(reservation_agent)
]


# ===========================================================================
# 8. Chat loop
#    - Keeps the conversation history so context carries over.
#    - Continues from whichever agent answered last; the first turn starts
#      from the Triage Agent.
#    - Catches guardrail tripwires and replies with a safe fallback.
# ===========================================================================
OFF_TOPIC_REPLY = (
    "저는 레스토랑 관련 질문에 대해서만 도와드리고 있어요. "
    "메뉴를 확인하거나, 예약하거나, 음식을 주문하실 수 있어요."
)
UNSAFE_OUTPUT_REPLY = (
    "죄송합니다. 방금은 적절한 답변을 드리지 못했어요. "
    "메뉴, 예약, 주문, 또는 불편하셨던 점에 대해 다시 말씀해 주시겠어요?"
)


async def main() -> None:
    print("🍽️  Restaurant bot here. How can I help?  (type 'quit' to exit)\n")

    history: list[dict] = []        # accumulated conversation input
    current_agent = triage_agent    # the agent currently in charge of replying

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThanks for stopping by!")
            break

        if user_input.lower() in {"quit", "exit", "종료"}:
            print("Thank you. Have a great day!")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        try:
            result = await Runner.run(current_agent, history)
        except InputGuardrailTripwireTriggered:
            # Off-topic or inappropriate input was blocked before the agent ran.
            print("Bot: [input guardrail triggered]")
            print(f"Bot: {OFF_TOPIC_REPLY}\n")
            history.append({"role": "assistant", "content": OFF_TOPIC_REPLY})
            continue
        except OutputGuardrailTripwireTriggered:
            # The agent produced an unprofessional / leaking reply; suppress it.
            print("Bot: [output guardrail triggered]")
            print(f"Bot: {UNSAFE_OUTPUT_REPLY}\n")
            history.append({"role": "assistant", "content": UNSAFE_OUTPUT_REPLY})
            continue

        print(f"{result.last_agent.name}: {result.final_output}\n")

        # Update history and current agent for the next turn.
        history = result.to_input_list()
        current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())