## Использование модуля moex_all_in_one_v2

Ниже — полная инструкция по использованию основной функции модуля: `process_log_v2`.
Документ описывает минимальный набор входных данных и полный (максимальный) сценарий с примерами и подсказками по отладке.

### Краткое описание

`process_log_v2` — это «всё-в-одном» сценарий для обработки логов торгов (trades log):

- загружает/кеширует OHLC (свечи) через внутренний загрузчик `moex_ohcl_loader`;
- гарантирует наличие справочника шагов (moex_tickers_steps.csv);
- вызывает сборщик equity (`build_equity`) для построения equity DataFrame;
- опционально: учитывает комиссию, строит `EquityReport` и рисует интерактивный график Plotly.

Функция возвращает `AllInOneResultV2` с полями: `log_path`, `candles_path`, `candles`, `equity`, `report`, `plot_figure`, `steps_path`.

### Зависимости

- Python 3.8+ (совместимо с окружением проекта)
- pandas
- plotly (только если хотите `show_plot=True`)
- остальные внутренние модули проекта (`data.moex_ohcl_loader`, `data.equity_builder`, `data.downloader_my` и т.д.)

Проект содержит `requirements.txt` — установите зависимости через pip в активном виртуальном окружении:

```powershell
python -m pip install -r requirements.txt
```

Если вы собираетесь использовать отображение графика, установите plotly:

```powershell
python -m pip install plotly
```

### Импорт и пример простого вызова (минимально необходимый)

Минимум: файл лога с записями торгов (CSV/текст), доступный по пути `log_path`.

Пример минимального сценария (без отчёта и без графика):

```python
from src.moex_all_in_one_v2 import process_log_v2

result = process_log_v2("path/to/trades_log.csv")

print(result.log_path)
print(result.candles.shape)
```

Требование: `log_path` должен указывать на существующий файл. В противном случае будет вызвано исключение `FileNotFoundError`.

### Полный список параметров `process_log_v2`

Все параметры с объяснениями (названия соответствуют сигнатуре функции в модуле):

- `log_path: PathLike` — путь к CSV/файлу логов торгов. Обязательный параметр.
- `datetime_column: str` — имя колонки с датой/временем в логе. Значение по умолчанию: `DEFAULT_DATETIME_COLUMN` (берётся из `moex_ohcl_loader`).
- `ticker_column: str` — имя колонки с тикером в логе. По умолчанию: `DEFAULT_TICKER_COLUMN`.
- `encoding: str` — кодировка файла лога. По умолчанию: `DEFAULT_LOG_ENCODING`.
- `steps_path: Optional[PathLike]` — путь к файлу `moex_tickers_steps.csv`. Если не передан, функция попытается найти стандартный файл рядом с модулями, затем — в `help_data`, и в крайнем случае вызовет внутренний загрузчик `downloader_my`.
- `help_data_dir: PathLike` — директория для вспомогательных данных (по умолчанию `help_data`). Эта директория будет создана при необходимости.
- `ohcl_dir: PathLike` — директория, в которую сохраняются (или из которой читаются) CSV со свечами (по умолчанию `OHCL`).
- `board: Optional[str]` — торговая площадка/борд (если требуется передать в загрузчик свечей). По умолчанию `None`.
- `margin_minutes: int` — сколько минут маржи учитывать при загрузке свечей вокруг трейдов. По умолчанию: `DEFAULT_MARGIN_MINUTES`.
- `buffer_minutes: int` — дополнительный буфер минут для загрузчика OHLC. По умолчанию: `DEFAULT_BUFFER_MINUTES`.
- `save_candles: bool` — сохранять ли полученные свечи в `OHCL/{logstem}_ohcl.csv`. По умолчанию `True`.
- `build_equity_report: bool` — строить ли `EquityReport` (использует `EquityReport` builder). По умолчанию `False`.
- `show_plot: bool` — строить ли интерактивный график (Plotly) по equity. По умолчанию `False`.
- `plot_column: str` — имя колонки equity, которую рисовать (например, `Equity`) — по умолчанию `"Equity"`.
- `comis: float` — комиссия, применяемая в сборщике equity (например 0.01 = 1%). По умолчанию `0.0`.

Примечание: точные имена значений по умолчанию (константы `DEFAULT_*`) определены в `data/moex_ohcl_loader.py` и `data/candle_loader.py`. Если вам важны конкретные значения — откройте соответствующие модули.

### Что делает функция — пошагово

1. Проверяет, что `log_path` существует.
2. Удостоверяется, что есть справочник шагов (шаги/типы тикеров) через `_ensure_steps_reference_v2`.
   - Если `steps_path` явно указан, проверяет его.
   - Иначе ищет `DEFAULT_STEPS_PATH` рядом с модулями.
   - Если не найден — использует `downloader_my` для получения копии в `help_data`.
3. Вызывает `download_moex_ohlc(...)` для получения DataFrame свечей (OHLC).
   - Свечи сохраняются в `ohcl_dir` и по умолчанию сохраняются в CSV `{log_stem}_ohcl.csv`.
4. При необходимости (если `build_equity_report` или `show_plot` = True) вызывает `build_equity(...)` для расчёта equity DataFrame.
   - Если `equity` пустой — функция предупреждает и не строит отчёт/график.
5. Если `build_equity_report` = True — вызывает `_build_equity_report(equity_df, ticker_info_csv=steps_reference)`.
6. Если `show_plot` = True — строит Plotly-figure с колоночной `plot_column`.

### Примеры использования

1) Минимальный (только загрузка свечей и возврат результата):

```python
from src.moex_all_in_one_v2 import process_log_v2

res = process_log_v2("logs/trades_2025-01-01.csv")
print(res.candles.head())
```

2) Полный сценарий: расчёт equity, отчёт и интерактивный график:

```python
from src.moex_all_in_one_v2 import process_log_v2

res = process_log_v2(
    "logs/trades_2025-01-01.csv",
    steps_path=None,                # позволить функции найти/скачать справочник шагов
    help_data_dir="help_data",
    ohcl_dir="OHCL",
    build_equity_report=True,
    show_plot=True,
    plot_column="Equity",
    comis=0.001                    # 0.1% комиссия
)

# Отчёт (если был построен)
if res.report is not None:
    print(res.report)

# Показать график (в jupyter или в окружении, где plotly поддерживается)
if res.plot_figure is not None:
    res.plot_figure.show()
```

3) Явно указать steps CSV (локальная копия):

```python
res = process_log_v2("log.csv", steps_path="help_data/moex_tickers_steps.csv")
```

### Выходные данные (`AllInOneResultV2`)

- `log_path: Path` — путь к входному файлу лога.
- `candles_path: Path` — путь к сохранённому CSV со свечами (OHCL) для этого лога.
- `candles: pd.DataFrame` — DataFrame со свечами, полученный от `download_moex_ohlc`.
- `equity: Optional[pd.DataFrame]` — DataFrame equity, если он был построен.
- `report: Optional[EquityReport]` — объект отчёта, если `build_equity_report=True`.
- `plot_figure: Optional[Any]` — Plotly Figure, если `show_plot=True`.
- `steps_path: Path` — путь к файлу шагов, который был использован.

### Обработка ошибок и подсказки по отладке

- FileNotFoundError: если `log_path` не найден — проверьте путь и кодировку.
- Если `steps` не найдены — функция попытается скачать через `downloader_my`. Убедитесь, что в окружении есть доступ и что `downloader_my` корректно настроен.
- RuntimeError при попытке построить график: если plotly не установлен — установите `plotly`.
- KeyError: если `plot_column` отсутствует в equity DataFrame — убедитесь, что вы указали корректное имя колонки (обычно `Equity`).

Логирование: модуль использует `logging.getLogger(__name__)`. Для подробного вывода включите уровень DEBUG/INFO в вашем приложении:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Часто встречающиеся проблемы

- "Свечи пустые" — возможные причины: некорректные имена колонок в логе (datetime/ticker), проблемы с `steps` (не соответствуют тикерам), или пустой диапазон времени в логе. Проверьте первые строки CSV: корректность формата даты и наличия тикера.
- "Equity пустой" — возможно, трансформатор ожидал другие столбцы (например, price/volume/side). Откройте `data/equity_builder.py` и `csv_transformer_profit` (если есть) для точного описания формата входного лога.

### Рекомендации и советы

- Подготовьте небольшой пример лога (10–20 строк) и прогоните `process_log_v2` локально для отладки.
- Если вы планируете часто перерасчитывать с разными параметрами — сохраняйте свечи (`save_candles=True`), чтобы не загружать их заново.
- Используйте `help_data` как место для единого справочника шагов в проекте, особенно при работе в команде.

### Примеры команд для Windows PowerShell

Установка зависимостей:

```powershell
# в активном виртуальном окружении
python -m pip install -r requirements.txt
python -m pip install plotly  # если нужен график
```

Запуск небольшого теста в интерактивной сессии Python:

```powershell
python -c "from src.moex_all_in_one_v2 import process_log_v2; print(process_log_v2('logs/some_log.csv'))"
```

### Заключение

Этот документ покрывает базовые и продвинутые сценарии использования `process_log_v2`. Если потребуется, можно расширить раздел: "Формат входного лог-файла" — добавив конкретный пример CSV со всеми необходимыми колонками (datetime, ticker, price, qty, side и т.д.) в зависимости от того, какие поля ожидает `data/equity_builder`.

Если нужно — могу дополнительно:

- добавить пример реального CSV-лога в `help_data/examples/`;
- сгенерировать unit-test, который демонстрирует минимальный рабочий пример;
- уточнить точные имена колонок и форматы дат, читая `data/equity_builder` и `data/moex_ohcl_loader`.

---

Файл создан автоматически как инструкция по использованию `process_log_v2`.
