# Методы оптимизации: код и отчёты

Учебные лабораторные работы по курсу **«Методы оптимизации»** в ИТМО.
Авторы отчётов: **Алексей Новиков и Александр Душкин**.

## С чего начать просмотр

Для первого знакомства удобно открыть [отчёт ЛР5: полиномиальная регрессия](lab5/Report_demo5.pdf).
В нём собраны постановка задачи, сравнение методов, исследование размера батча
и регуляризации, графики и таблицы экспериментов. Для просмотра PDF запускать код не нужно.

| Работа | Тема | Материалы |
| --- | --- | --- |
| ЛР1 | Одномерная минимизация | [Код](lab1/) · [Исторический отчёт исходной реализации](legacy/lab1_original/Report.pdf) |
| ЛР2 | Градиентный спуск и выбор шага | [Отчёт PDF](lab2/Report_demo2.pdf) · [Код и конфигурация](lab2/) |
| ЛР3 | Momentum и адаптивные методы | [Отчёт PDF](lab3/Report_demo3.pdf) · [Код и конфигурация](lab3/) |
| ЛР4 | Сопряжённые градиенты, Ньютон и квазиньютоновские методы | [Отчёт PDF](lab4/Report_demo4.pdf) · [Код и конфигурация](lab4/) |
| ЛР5 | Полиномиальная регрессия и регуляризация | [Отчёт PDF](lab5/Report_demo5.pdf) · [Код и конфигурация](lab5/) |

Дополнительный обзор: [слайды ЛР2–3](presentation/presentation_labs_2_3.pdf)
и [слайды ЛР4–5](presentation/presentation_labs_4_5.pdf).
Полные постановки и результаты находятся в отчётах.

Это совместные учебные работы, а не публикация нового метода.

Проект разделён на переиспользуемую библиотеку `optimlib` и отдельные директории
лабораторных работ. Лабораторные импортируют ядро библиотеки и содержат только
свои функции, конфиги и скрипты запуска.

## Структура

```text
.
+-- optimlib/          # библиотека: ядро, оптимизаторы, эксперименты, графики
+-- lab1/              # лабораторная 1: одномерный поиск
+-- lab2/              # лабораторная 2: классические градиентные методы
+-- lab3/              # лабораторная 3: momentum и адаптивные методы
+-- lab4/              # лабораторная 4: сопряжённые направления и методы Ньютона
+-- lab5/              # лабораторная 5: задача регрессии
+-- legacy/            # старая версия первой лабораторной для сверки
+-- README.md
```

## Установка

Из корня репозитория установите библиотеку в режиме разработки:

```powershell
.\.venv\Scripts\pip.exe install -e .\optimlib[dev]
```

Если виртуальное окружение ещё не создано:

```powershell
py -m venv .venv
.\.venv\Scripts\pip.exe install -e .\optimlib[dev]
```

## Запуск лабораторных

```powershell
.\.venv\Scripts\python.exe lab1\run.py
.\.venv\Scripts\python.exe lab2\run.py
.\.venv\Scripts\python.exe lab3\run.py
.\.venv\Scripts\python.exe lab4\run.py
.\.venv\Scripts\python.exe lab5\run.py
```
Лабораторная 1 сохраняет результаты в `outputs/lab1`. В ней реализованы методы
одномерной минимизации на отрезке: пассивный поиск, дихотомия, золотое сечение,
метод Фибоначчи, параболическая интерполяция и обёртка SciPy Brent. В конфиге
сравниваются унимодальные тестовые функции и мультимодальная функция Растригина.

Лабораторная 2 сохраняет результаты в `outputs/lab2`. В ней реализованы
классические градиентные методы для многомерных функций: градиентный спуск с
постоянным шагом, с дроблением шага (с условием Армихо), с дроблением шага (с сильными условиями Вольфе) и наискорейший
спуск с одномерным поиском. Для тестов используются квадратичные функции,
Розенброка, Экли и Химмельблау; строятся линии уровня, траектории и графики
сходимости.

Лабораторная 3 сохраняет результаты в `outputs/lab3`. В ней реализованы
momentum и адаптивные градиентные методы: Momentum, Nesterov, AdaGrad, RMSProp,
AdaDelta и Adam. Конфиг содержит сетки гиперпараметров, а запуск строит heatmap
чувствительности и траектории лучших запусков для выбранных функций.

Лабораторная 4 сохраняет результаты в `outputs/lab4`. В ней реализованы:
квадратичный метод сопряжённых градиентов, нелинейные CG Флетчера-Ривса и
Полака-Рибьера, Ньютона с использованием разложения Холецкого, Ньютона с выбором направления, Powell Dog Leg,
DFP, BFGS, L-BFGS и сравнение со `scipy.optimize.minimize(method="Newton-CG")`.

Лабораторная 5 сохраняет результаты в `outputs/lab5`. Реализованы линейная и
полиномиальная регрессия степеней 1-5, варианты без регуляризации, L1, L2 и
Elastic Net. Методы оптимизации: аналитическое решение одномерной линейной
регрессии, SGD, mini-batch GD, Гаусса–Ньютона и Гаусса–Ньютона с регуляризацией Левенберга–Марквардта.

## Как добавить новый метод оптимизации

1. Выберите подходящий уровень абстракции. Для одномерного поиска используйте
   `IntervalOptimizer`, для классических градиентных методов можно наследоваться
   от `GradientOptimizer`, для методов со своей логикой достаточно реализовать
   интерфейс `Optimizer.minimize(func, config)`.
2. Метод должен принимать `ObjectiveFunction` и `OptimizerConfig`, возвращать
   `OptimizationResult`, а траекторию записывать через `StepState`.
3. Зарегистрируйте метод через `register_optimizer("name", OptimizerClass)`.
4. Если методу нужны параметры, добавьте явные поля в `OptimizerConfig`, чтобы
   конфиг был типобезопасным и проверялся mypy.
5. Для графиков добавляйте общий код в `visualization/convergence.py`,
   `contour.py` или `base.py`; лабораторно-специфичный код держите в отдельном
   модуле вроде `visualization/lab4.py`.
6. Добавьте быстрые тесты в `optimlib/tests`: сходимость на простой функции,
   корректность истории и отсутствие численных регрессий.

Минимальный пример:

```python
from optimlib.core.base import ObjectiveFunction, OptimizationResult, StepState
from optimlib.core.callbacks import HistoryCallback
from optimlib.core.config import OptimizerConfig
from optimlib.functions.base import MultivariateFunction
from optimlib.utils.registry import register_optimizer


class MyGradientMethod:
    name = "MyGradientMethod"

    def minimize(self, func: ObjectiveFunction, config: OptimizerConfig) -> OptimizationResult:
        if not isinstance(func, MultivariateFunction):
            raise TypeError("MyGradientMethod requires a MultivariateFunction.")
        history = HistoryCallback()
        history.on_start()
        x = func.initial_point()
        f_value = func(x)
        grad = func.gradient(x)
        for iteration in range(config.max_iter):
            x = x - config.alpha * grad
            f_value = func(x)
            grad = func.gradient(x)
            state = StepState(iteration, x, f_value, grad, config.alpha)
            history.on_step(state)
            if float((grad @ grad) ** 0.5) <= config.tol_grad:
                break
        return OptimizationResult(
            x=x,
            f=f_value,
            n_iter=len(history.history),
            n_calls=func.call_count,
            n_grad_calls=func.grad_count,
            converged=True,
            message="finished",
            history=history.history,
            metadata={"optimizer": self.name},
        )


register_optimizer("my_gradient_method", MyGradientMethod)
```

## Проверки

```powershell
cd optimlib
..\.venv\Scripts\python.exe -m pytest
..\.venv\Scripts\python.exe -m mypy src ..\lab1 ..\lab2 ..\lab3 ..\lab4 ..\lab5
```

## Компиляция `.tex` файлов

Нужна установленная TeX-система с `pdflatex` и поддержкой русского языка.
Для сборки презентаций из корня репозитория:

```powershell
.\presentation\build.ps1
```

Скрипт выполняет два прохода для корректного общего числа слайдов,
сохраняет служебные файлы в `presentation/.build/`, а PDF — рядом с `.tex`.
Графики презентаций сохранены в `presentation/assets/`: это изображения из
исходных PDF, поэтому оформление можно пересобрать без повторных экспериментов
и без подмены графиков результатами более поздних запусков.
Новые результаты лабораторных по-прежнему сохраняются в `outputs/`.

Пример сборки отчёта после генерации результатов лабораторной:

```powershell
Push-Location lab5
pdflatex -interaction=nonstopmode -halt-on-error Report_demo5.tex
pdflatex -interaction=nonstopmode -halt-on-error Report_demo5.tex
Pop-Location
```
