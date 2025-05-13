import openpyxl

class ExcelParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_test_data(self):
        workbook = openpyxl.load_workbook(self.file_path)
        sheet = workbook.active
        data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            data.append(row)
        return data
