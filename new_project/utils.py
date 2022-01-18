import json
import os


class Logger:
    def __init__(self, args):
        self.base = "results"
        os.makedirs(self.base, exist_ok=True)
        os.makedirs(f"{self.base}/{args.criterion}", exist_ok=True)
        os.makedirs(f"{self.base}/{args.criterion}/{args.aug_method}", exist_ok=True)

        self.folder_name = f"{args.criterion}/{args.aug_method}"
        self.path = f"{self.base}/{self.folder_name}"
        self.log_file_name = self.path + "/log.txt"
        self.log_file = open(self.log_file_name, "w")

    def __call__(self, item):
        print(item)
        print(item, file=self.log_file)

    def write(self, data):
        with open(f"{self.path}/results.json", "w") as f:
            json.dump(data, f, default=str)

