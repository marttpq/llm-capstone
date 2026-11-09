import unittest
from fastapi.testclient import TestClient
from main import app


class TestFitnessAssistantAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes the FastAPI TestClient and triggers startup event."""
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        """Cleanly exits the TestClient."""
        cls.client.__exit__(None, None, None)

    def test_01_health_check(self):
        """Verifies root endpoint returns 200 OK and serves the HTML frontend."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("Fitness LLM Agent", response.text)

    def test_02_vector_search_endpoint(self):
        """Verifies /search endpoint returns relevant knowledge chunks with citations."""
        response = self.client.get("/search?query=เจ็บข้อมือ&top_k=2")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("query"), "เจ็บข้อมือ")
        self.assertEqual(data.get("top_k"), 2)
        results = data.get("results", [])
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("citation", results[0])
        self.assertIn("score", results[0])

    def test_03_search_empty_query_validation(self):
        """Verifies /search endpoint returns 400 when query is whitespace/empty."""
        response = self.client.get("/search?query=   ")
        self.assertEqual(response.status_code, 400)

    def test_04_agent_chat_endpoint(self):
        """Verifies /chat endpoint executes the ReAct agent loop and returns an answer."""
        response = self.client.post(
            "/chat",
            json={"message": "เจ็บข้อศอกเล่นท่าไหนแทน Barbell Curl ได้บ้าง"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertGreater(len(data["response"]), 10)

    def test_05_chat_empty_message_validation(self):
        """Verifies /chat endpoint returns 400 when message is empty."""
        response = self.client.post("/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
