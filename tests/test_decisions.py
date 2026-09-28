import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

from decisions import ask
from decisions.answers import (above_uniform, answer, entropy_confidence, options, probabilities_of,
                               read_answers, top_probability)
from decisions.calibration import (brier_score, coverage_and_accuracy, decide, expected_calibration_error,
                                   fit_temperature, softmax, synthetic_router)
from decisions.questions import TICKET, TRIAGE_QUESTIONS, check_request, choice, noul, question_problems, \
    request_body, score


def fake_response(questions, model="fake-1"):
    answers = {}
    for question_id, question in questions.items():
        count = len(options(question))
        if question["type"] == "choice":
            p = [0.9] + [0.1 / (count - 1)] * (count - 1)
        elif question["type"] == "score":
            p = [0.2, 0.3, 0.5]
        else:
            p = [0.8, 0.2]
        reply = answer(question, p)
        if question["type"] != "noul":
            reply["confidence"] = 0.5
        answers[question_id] = reply
    return {"model": model, "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}


class FakeSystemOne(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        FakeSystemOne.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if self.path != "/v1/systemone":
            return self.reply(404, {"detail": "not found"})
        if body["state"] == "redirect me":
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if body["state"] == "invalid":
            return self.reply(422, {"detail": "questions.department: bad"})
        if body["state"] == "broken":
            return self.reply(200, {"model": "fake-1", "usage": {}})
        return self.reply(200, fake_response(body["questions"]), {"x-typesafe-request-id": "req-1"})

    def reply(self, status, payload, headers=None):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def run(argv):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = ask.main(argv)
    return result, output.getvalue()


class QuestionTests(unittest.TestCase):
    def test_course_request_is_valid_and_ordered(self):
        body = request_body(TICKET)
        self.assertEqual(list(body), ["model", "state", "questions"])
        self.assertEqual(check_request(body), [])
        self.assertEqual(noul("Is it spam?"), {"type": "noul", "instructions": "Is it spam?"})
        self.assertEqual(score("Rate", ["low", "high"])["criteria"], ["low", "high"])

    def test_structural_problems_are_errors(self):
        broken = [
            {"type": "noul"},
            {"type": "noul", "instructions": "Spam?", "criteria": {"maybe": "unsure"}},
            choice("Pick", {"only": None}),
            score("Rate", ["one level"]),
            {"type": "rank", "instructions": "Order these"},
            "not a question",
        ]
        for question in broken:
            with self.subTest(question=question):
                self.assertTrue(question_problems(question)[0])
        for body in ({"model": "", "state": "x", "questions": TRIAGE_QUESTIONS},
                     {"model": "m", "state": " ", "questions": TRIAGE_QUESTIONS},
                     {"model": "m", "state": "x", "questions": {}}):
            with self.assertRaises(ValueError):
                check_request(body)

    def test_style_problems_are_warnings(self):
        errors, warnings = question_problems(choice("Spam?", {"Yes": None, "no": None}))
        self.assertEqual(errors, [])
        self.assertIn("booleans", warnings[0])
        many = choice("Which product?", {f"product_{i}": None for i in range(25)})
        self.assertIn("token budget", question_problems(many)[1][0])


class AnswerTests(unittest.TestCase):
    def test_answers_follow_the_wire_shapes(self):
        department = answer(TRIAGE_QUESTIONS["department"], [0.1, 0.7, 0.1, 0.1])
        self.assertEqual(department["choice"], "account")
        self.assertEqual(list(department["probabilities"]), ["billing", "account", "technical", "other"])
        urgency = answer(TRIAGE_QUESTIONS["urgency"], [0.2, 0.3, 0.5])
        self.assertAlmostEqual(urgency["score"], 1.3)
        self.assertEqual(list(urgency["legend"]), ["0", "1", "2"])
        self.assertEqual(answer(TRIAGE_QUESTIONS["refund_requested"], [0.25, 0.75]),
                         {"type": "noul", "noul": 0.75})
        for bad in ([0.5, 0.5], [0.5, 0.6, -0.1, 0.0], [0.4, 0.4, 0.1, 0.0]):
            with self.assertRaises(ValueError):
                answer(TRIAGE_QUESTIONS["department"], bad)

    def test_confidence_definitions(self):
        for n in (2, 4, 10):
            uniform, certain = np.full(n, 1 / n), np.eye(n)[0]
            self.assertAlmostEqual(above_uniform(uniform), 0.0)
            self.assertAlmostEqual(entropy_confidence(uniform), 0.0)
            self.assertAlmostEqual(above_uniform(certain), 1.0)
            self.assertAlmostEqual(entropy_confidence(certain), 1.0)
        p = [0.7, 0.1, 0.1, 0.1]
        self.assertAlmostEqual(top_probability(p), 0.7)
        self.assertAlmostEqual(above_uniform(p), 0.6)
        expected = 1 + sum(value * np.log(value) for value in p) / np.log(4)
        self.assertAlmostEqual(entropy_confidence(p), expected)
        self.assertAlmostEqual(entropy_confidence(p), 0.3216, places=4)
        padded = [above_uniform([0.6] + [0.4 / (n - 1)] * (n - 1)) for n in (2, 20)]
        np.testing.assert_allclose(padded, [0.2, 11 / 19])
        self.assertAlmostEqual(top_probability(probabilities_of({"type": "noul", "noul": 0.3})), 0.7)

    def test_read_answers_accepts_both_backends(self):
        response = fake_response(TRIAGE_QUESTIONS)
        response["answers"]["department"]["answer_confidence"] = 0.9
        read = read_answers(response, TRIAGE_QUESTIONS)
        self.assertEqual(read["department"]["answer"], "billing")
        self.assertAlmostEqual(read["department"]["top_probability"], 0.9)
        self.assertEqual(read["department"]["answer_confidence"], 0.9)
        self.assertIsNone(read["refund_requested"]["reported_confidence"])
        self.assertAlmostEqual(read["urgency"]["answer"], 1.3)

    def test_read_answers_rejects_mismatches(self):
        def broken(change):
            response = fake_response(TRIAGE_QUESTIONS)
            change(response["answers"])
            return response

        cases = {
            "missing": broken(lambda a: a.pop("urgency")),
            "wrong type": broken(lambda a: a["urgency"].update(type="choice")),
            "unknown label": broken(lambda a: a["department"].update(choice="sales")),
            "missing probability": broken(lambda a: a["department"]["probabilities"].pop("other")),
            "noul range": broken(lambda a: a["refund_requested"].update(noul=1.5)),
            "score range": broken(lambda a: a["urgency"].update(score=3.0)),
            "sum": broken(lambda a: a["urgency"]["probabilities"].update({"2": 0.9})),
        }
        for name, response in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                read_answers(response, TRIAGE_QUESTIONS)
        with self.assertRaises(ValueError):
            read_answers({"detail": "error"}, TRIAGE_QUESTIONS)


class CalibrationTests(unittest.TestCase):
    def test_metrics_on_known_cases(self):
        confidence = np.full(10, 0.8)
        correct = np.array([1] * 8 + [0] * 2)
        self.assertAlmostEqual(expected_calibration_error(confidence, correct), 0.0)
        self.assertAlmostEqual(expected_calibration_error([1.0, 1.0], [0, 0]), 1.0)
        self.assertAlmostEqual(brier_score(np.eye(3), [0, 1, 2]), 0.0)
        self.assertAlmostEqual(brier_score(np.full((1, 3), 1 / 3), [0]), 2 / 3)
        self.assertEqual(coverage_and_accuracy([0.2, 0.4], [1, 0], 0.9), (0.0, None))
        self.assertEqual(coverage_and_accuracy([0.95, 0.4], [1, 0], 0.9), (0.5, 1.0))

    def test_temperature_recovers_known_overconfidence(self):
        logits, labels = synthetic_router(np.random.default_rng(0), 3000, overconfidence=2.0)
        temperature = fit_temperature(logits, labels)
        self.assertLess(abs(temperature - 2.0), 0.3)
        np.testing.assert_array_equal(softmax(logits).argmax(axis=1),
                                      softmax(logits, temperature).argmax(axis=1))

    def test_bands(self):
        self.assertEqual([decide(value) for value in (0.95, 0.9, 0.7, 0.2)],
                         ["act", "act", "confirm", "human"])
        with self.assertRaises(ValueError):
            decide(0.5, act_at=0.5, confirm_at=0.8)


class AskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSystemOne)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        FakeSystemOne.seen.clear()

    def test_urls_must_be_https_or_loopback(self):
        self.assertEqual(ask.server_url("https://gateway.example/api/"), "https://gateway.example/api")
        self.assertEqual(ask.server_url("http://[::1]:8000"), "http://[::1]:8000")
        for url in ("http://example.com", "http://127.0.0.1.evil.example", "https://" + "user" + "@example.com",
                    "https://example.com?key=1", "https://example.com#x", "ftp://example.com",
                    "http://127.0.0.1:99999"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ask.server_url(url)

    def test_dry_runs_contact_nothing(self):
        refuse = mock.Mock(side_effect=AssertionError("network used in a dry run"))
        with mock.patch.object(ask, "post", refuse), mock.patch.object(ask, "LayaInProcess", refuse):
            for argv in ([], ["--jev"], ["--url", self.url, "--examples"], ["--laya", "--allow-download"]):
                with self.subTest(argv=argv):
                    result, output = run(argv)
                    self.assertIsNone(result)
                    self.assertIn("Dry run", output)
        self.assertEqual(FakeSystemOne.seen, [])

    def test_jev_requires_a_key_from_the_environment(self):
        refuse = mock.Mock(side_effect=AssertionError("request sent without a key"))
        with mock.patch.dict(os.environ), mock.patch.object(ask, "post", refuse):
            os.environ.pop("TYPESAFE_API_KEY", None)
            with self.assertRaises(SystemExit) as caught:
                run(["--jev", "--live"])
        self.assertIn("TYPESAFE_API_KEY", str(caught.exception))

    def test_live_request_sends_key_only_when_asked(self):
        with mock.patch.dict(os.environ, {"COURSE_TEST_KEY": "test-key"}):
            report, output = run(["--url", self.url, "--live", "--api-key-env", "COURSE_TEST_KEY"])
        self.assertEqual(FakeSystemOne.seen[0]["headers"]["Authorization"],
                         " ".join((ask.AUTH_SCHEME, "test-key")))
        self.assertEqual(FakeSystemOne.seen[0]["body"], request_body(TICKET))
        self.assertEqual(report["records"][0]["request_id"], "req-1")
        self.assertEqual(report["records"][0]["answers"]["department"]["answer"], "billing")
        self.assertIn("Answered by: fake-1", output)
        run(["--url", self.url, "--live"])
        self.assertNotIn("Authorization", FakeSystemOne.seen[1]["headers"])

    def test_examples_are_scored_and_saved(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {"COURSE_TEST_KEY": "s3cr3t"}):
            path = Path(folder) / "runs" / "fake.json"
            argv = ["--url", self.url, "--live", "--examples", "--api-key-env", "COURSE_TEST_KEY",
                    "--save", str(path)]
            report, output = run(argv)
            saved = path.read_text(encoding="utf-8")
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    run(argv)
        self.assertNotIn("s3cr3t", saved)
        self.assertEqual(json.loads(saved)["summary"], report["summary"])
        summary = report["summary"]
        self.assertEqual(summary["examples"], 15)
        department, refund = summary["questions"]["department"], summary["questions"]["refund_requested"]
        self.assertAlmostEqual(department["accuracy"], 5 / 15)
        self.assertEqual(department["bands"]["act"], {"count": 15, "correct": 5})
        self.assertAlmostEqual(refund["accuracy"], 12 / 15)
        self.assertEqual(refund["bands"]["confirm"], {"count": 15, "correct": 12})
        self.assertIsNone(department["ece"])
        self.assertEqual(summary["input_tokens"], 150)
        self.assertIn("ECE is not reported", output)

    def test_server_problems_stop_the_run(self):
        for state, message in (("redirect me", "HTTP 302"), ("invalid", "HTTP 422"),
                               ("broken", "Unexpected response")):
            with self.subTest(state=state):
                with self.assertRaises(SystemExit) as caught:
                    run(["--url", self.url, "--live", "--ticket", state])
                self.assertIn(message, str(caught.exception))
        self.assertNotIn("/elsewhere", [request["path"] for request in FakeSystemOne.seen])

    def test_bundled_examples_and_label_checks(self):
        examples = ask.load_examples(ask.EXAMPLES, TRIAGE_QUESTIONS)
        self.assertEqual(len(examples), 15)
        self.assertTrue(all(set(example["labels"]) == {"department", "refund_requested"} for example in examples))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            for row in ({"id": "a", "state": "x", "labels": {"department": "sales"}},
                        {"id": "a", "state": "x", "labels": {"refund_requested": "yes"}},
                        {"id": "a", "state": "x", "labels": {"urgency": 3}},
                        {"id": "a", "state": "x", "labels": {"unknown": 1}}):
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.subTest(row=row), self.assertRaises(ValueError):
                    ask.load_examples(path, TRIAGE_QUESTIONS)

    def test_laya_stays_offline_unless_downloads_are_allowed(self):
        created = []

        class Router:
            def __init__(self, **kwargs):
                created.append({"offline": os.environ.get("HF_HUB_OFFLINE"), **kwargs})

            def predict(self, state, questions, model=None):
                response = fake_response(questions, model="laya-rl-agent")
                response["routing"] = {"model": "english", "reason": "detected English"}
                return response

        fake_laya = types.ModuleType("laya")
        fake_laya.Router, fake_laya.__version__ = Router, "test"
        with mock.patch.dict(sys.modules, {"laya": fake_laya}), mock.patch.dict(os.environ):
            os.environ.pop("HF_HUB_OFFLINE", None)
            report, output = run(["--laya", "--live"])
            self.assertEqual(created[-1], {"offline": "1"})
            os.environ.pop("HF_HUB_OFFLINE", None)
            run(["--laya", "--live", "--allow-download", "--revision", "reviewed"])
            self.assertEqual(created[-1], {"offline": None, "revision": "reviewed"})
        self.assertIn("routed to english", output)
        self.assertEqual(report["laya"], {"version": "test", "revision": None, "downloads_allowed": False})

    def test_missing_laya_explains_the_install(self):
        with mock.patch.dict(sys.modules, {"laya": None}), mock.patch.dict(os.environ):
            with self.assertRaises(SystemExit) as caught:
                run(["--laya", "--live"])
        self.assertIn("pip install laya", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
