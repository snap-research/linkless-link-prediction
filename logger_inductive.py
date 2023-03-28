import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 5
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'  Final val: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 1]:.2f}')
            print(f'   old_old Test: {result[argmax, 2]:.2f}')
            print(f'   old_new Test: {result[argmax, 3]:.2f}')
            print(f'   new_new Test: {result[argmax, 4]:.2f}')
        else:
            best_results = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                val = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 1].item()
                old_old = r[r[:, 1].argmax(), 2].item()
                old_new = r[r[:, 1].argmax(), 3].item()
                new_new = r[r[:, 1].argmax(), 4].item()
                best_results.append((val, test, old_old, old_new, new_new))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'  Final val: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'   Final old_old: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final old_new: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final new_new: {r.mean():.2f} ± {r.std():.2f}')


