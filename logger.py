import csv

class Logger:
    def __init__(self, str_format, stat_names, interval, csv_dir, epoch_batch_str_format="=== Epoch {} ({:2.1f}%) ===\n", write_header=True):
        self.stat_names = stat_names
        self.stats = {}
        for name in stat_names:
            self.stats[name] = 0.0

        self.interval = interval
        self.str_format = epoch_batch_str_format + str_format

        self.f = open(csv_dir, "a")
        self.csv_writer = csv.writer(self.f)
        if write_header:
            self.csv_writer.writerow(["Epoch", "Batch"] + stat_names)
        self.f.flush()

    def average(self):
        for name in self.stats:
            self.stats[name] /= self.interval

    def reset_stats(self):
        for name in self.stats:
            self.stats[name] = 0.0

    def log(self, epoch, epoch_percent):
        self.average()
        ordered_stats = [epoch, epoch_percent] + [self.stats[n] for n in self.stat_names]

        print(self.str_format.format(*ordered_stats))
        self.csv_writer.writerow(ordered_stats)
        self.f.flush()

        self.reset_stats()

    def close(self):
        self.f.close()
