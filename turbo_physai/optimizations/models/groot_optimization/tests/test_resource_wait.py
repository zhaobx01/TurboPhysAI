import json

from groot_optimization.resource_wait import wait_for_idle_devices


def sample(percent=1):
    return {f'card{i}': {'vram Total Used Memory (B)': percent,
                        'vram Total Memory (B)': 100} for i in range(8)}


def test_requires_three_consecutive_samples_and_sleeps_between_checks(tmp_path):
    responses = iter([sample(1), sample(2), sample(3)])
    sleeps = []
    log = tmp_path / 'wait.jsonl'
    wait_for_idle_devices(log, query=lambda: next(responses), sleep=sleeps.append)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row['consecutive_idle'] for row in rows] == [1, 2, 3]
    assert sleeps == [10, 10]


def test_busy_or_failed_check_resets_consecutive_count(tmp_path):
    def response():
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    responses = iter([sample(1), sample(5), sample(1), RuntimeError('unavailable'),
                      sample(1), sample(1), sample(1)])
    sleeps = []
    log = tmp_path / 'wait.jsonl'
    wait_for_idle_devices(log, query=response, sleep=sleeps.append)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row['consecutive_idle'] for row in rows] == [1, 0, 1, 0, 1, 2, 3]
    assert len(sleeps) == 6 and all(delay == 10 for delay in sleeps)


def test_missing_card_or_invalid_total_cannot_count_as_idle(tmp_path):
    incomplete = sample()
    del incomplete['card7']
    invalid = sample()
    invalid['card4']['vram Total Memory (B)'] = 0
    responses = iter([sample(), incomplete, sample(), invalid, sample(), sample(), sample()])
    log = tmp_path / 'wait.jsonl'
    wait_for_idle_devices(log, query=lambda: next(responses), sleep=lambda _: None)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row['consecutive_idle'] for row in rows] == [1, 0, 1, 0, 1, 2, 3]


def test_mib_output_with_single_busy_card_resets_count(tmp_path):
    cards = {f'card{i}': {'vram Total Used Memory (MiB)': '1',
                          'vram Total Memory (MiB)': '100'} for i in range(8)}
    busy = {key: dict(value) for key, value in cards.items()}
    busy['card7']['vram Total Used Memory (MiB)'] = '5'
    responses = iter([cards, busy, cards, cards, cards])
    log = tmp_path / 'wait.jsonl'
    wait_for_idle_devices(log, query=lambda: next(responses), sleep=lambda _: None)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row['consecutive_idle'] for row in rows] == [1, 0, 1, 2, 3]


def test_reports_total_wait_time_without_resetting_after_busy_sample(tmp_path, capsys):
    times = iter([100.0, 100.0, 130.0, 160.0, 190.0, 220.0])
    responses = iter([sample(1), sample(5), sample(1), sample(1), sample(1)])
    log = tmp_path / 'wait.jsonl'
    wait_for_idle_devices(log, query=lambda: next(responses), sleep=lambda _: None,
                          clock=lambda: next(times))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row['elapsed_seconds'] for row in rows] == [0, 30, 60, 90, 120]
    assert '已等待 2分00秒' in capsys.readouterr().out
