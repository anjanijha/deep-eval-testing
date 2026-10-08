from app.llm_client import LLMClient


class RAGApplication:

    def __init__(self):

        self.llm = LLMClient()

        self.knowledge_base = [
            "Customers can request a refund within 30 days of purchase.",
            "The product must be unused and the original receipt is required.",
            "Customers can contact support through email or phone.",
            "Credit cards, debit cards and UPI are supported payment methods.",
            "Customers can reset their password using the password reset link.",
        ]

    def retrieve(self, question: str) -> list[str]:

        question = question.lower()

        context = []

        if "refund" in question:
            context.extend([self.knowledge_base[0], self.knowledge_base[1]])

        elif "support" in question:
            context.append(self.knowledge_base[2])

        elif "payment" in question:
            context.append(self.knowledge_base[3])

        elif "password" in question:
            context.append(self.knowledge_base[4])

        return context

    def generate_answer(self, question: str, context: list[str]) -> str:

        context_text = "\n".join(context)

        prompt = f"""
You are a customer support assistant.

Answer the question using ONLY the provided context.

Context:
{context_text}

Question:
{question}

Rules:
- Do not invent information.
- If the context does not contain the answer, say:
  "I don't have enough information."
"""

        return self.llm.generate(prompt)

    def ask(self, question: str) -> dict:

        context = self.retrieve(question)

        answer = self.generate_answer(question, context)

        return {"question": question, "answer": answer, "context": context}


rag_app = RAGApplication()


def ask(question: str) -> dict:
    return rag_app.ask(question)
