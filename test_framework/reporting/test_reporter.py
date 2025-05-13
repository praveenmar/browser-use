import json
import os

class TestReporter:
    def __init__(self):
        self.logs = []

    def log_action(self, action, result):
        self.logs.append({
            "action": action,
            "result": result
        })

    def log_validation(self, validation_action, result):
        self.logs.append({
            "validation": validation_action,
            "result": result
        })

    def generate_report(self):
        os.makedirs("reports", exist_ok=True)
        with open("reports/test_report.json", "w") as report_file:
            json.dump(self.logs, report_file, indent=4)
