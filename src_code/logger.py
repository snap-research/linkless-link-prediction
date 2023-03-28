import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def reset(self, run):
        assert run >= 0 and run < len(self.results)
        self.results[run] = []

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {result[:, 0].max():.4f}')
            print(f'   Final Test: {result[argmax, 1]:.4f}')
        else:
            best_results = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            # result = torch.tensor(self.results)

            # best_results = []
            # for r in result:
            #     valid = r[:, 0].max().item()
            #     test = r[r[:, 0].argmax(), 1].item()
            #     best_results.append((valid, test))

            # best_result = torch.tensor(best_results)
            # print(best_result)

            # print(f'All runs:')
            # r = best_result[:, 0]
            # print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            # r = best_result[:, 1]
            # print(f'   Final Test: {r.mean():.4f} ± {r.std():.4f}')