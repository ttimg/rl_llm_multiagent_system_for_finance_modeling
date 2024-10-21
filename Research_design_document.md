# Research Design Doc

## Дизайн исследования многоагентные системы, основанной на больших языковых моделях и методах обучения с подкреплением для принятия финансовых решений и моделирования бизнес-циклов

## Аннотация нализировать нализировать сложные экономические процессы, взаимодействия и принятие решений.
В данном документе разбирается исследование поведения участников фондового рынка, процессы их взаимодействия и принятия решения с использованием методов обучения с подкреплением (Reinforcement Learning, RL) на базе многоагентной системы больших языковых моделей (Large Language models, LLM). Рассматриваются ключевые макроэкономические эффекты: Эффективное распределение ресурсов, ожидание инфляции, рост и падение фондового рынка.
Также описан подробный план разработки данный системы от сбора данных до тестирования RL - алгоритмов. Архитектура системы обеспечивает эффективность торговых стратегий в условиях неопределенности. Коммуникации агентов структурированы таким образом, чтобы способствовать координации действий, обмену информацией, конкуренции и коллективному обучению, что повышает общую производительность и устойчивость системы. 
Система продемонстрировала сильные возможности обобщения в финансовых задачах, включая торговлей ценными бумаги и портфельным менеджментом 

## Введение
Сложности и колебания финансовых рынков создают существенные проблемы для принятия высококачественных, последовательных инвестиционных решений.
В таких задачах, как торговля ценными бумагами и управлением портфелем, каждое разумное решение основывается на множественных рыночных взаимодействиях и интегрированной информации, характеризующейся различными уровнями своевременности и методов. 
Эти задачи направлены на максимизацию прибыли при одновременном управлении текущими рыночными рисками в открытой среде.

На практике финансовые организации часто полагаются на синтезированную человеческую командную работу с организационной структурой, которая обычно включает в себя иерархическое общение между различными функциональными ролями, такими как аналитики данных, аналитики рисков, портфельные менеджеры и т. д., 
и тщательную интеграцию различных ресурсов. Однако когнитивные ограничения членов человеческой команды могут ограничить их способность быстро обрабатывать рыночные сигналы и достигать оптимальных инвестиционных результатов.

Для повышения прибыли от инвестиций и преодоления человеческих ограничений в исследованиях использовались методы RL, разработалась многоагентная система, которая имитируют рыночную среду и автоматизирует инвестиционные стратегии. Между тем, достижения в области больших языковых моделей (LLM) демонстрируют потенциал в сложных задачах, таких как рассуждения, 
использование инструментов, планирование и принятие решений, обещая превзойти существующие архитектуры агентов. Языковые агенты известны общением, подобным человеку, и адаптируемыми структурами, основанными на подсказках, предназначенными для принятия решений. 
Для оптимальной эффективности принятия решений необходимо учитывать два ключевых элемента: 
- Организационные агенты для содействия эффективной командной работе и эффективной коммуникации
- Дать агентам возможность постоянно учиться и совершенствовать свои действия.

Исследования показывают, что имитация организационных структур человека может эффективно координировать языковых агентов для конкретных задач. Кроме того, недавние разработки в оптимизации подсказки на основе текстового градиента и вербального подкрепления оказались практическими для итеративного улучшения рассуждений и возможностей принятия решений языковых агентов
Системы языковых агентов, адаптированные для принятия финансовых решений, такие как FINGPT, FINMEM и FINAGENT, продемонстрировали впечатляющую производительность. Тем не менее, эти системы имеют несколько ограничений:
- Во-первых, их зависимость от рисковых преференций агентов, основанных на краткосрочных колебаниях рынка, не контролирует общую подверженность риску в течение долгосрочных торговых периодов. Этот надзор может привести к потенциальным потерям из-за отклонения фундаментальных факторов, способствующих инвестиционной прибыли. Более эффективной стратегией является количественная оценка инвестиционных рисков с использованием мер риска из количественного финансирования.
- Во-вторых, большинство существующих систем предназначены для задач торговли одним активом и демонстрируют ограниченную адаптивность к другим финансовым приложениям, включающим несколько активов, таких как управление портфелем.
- В-третьих, эти системы зависят от понимания информации и способности одного агента к пониманию и извлечения, оказывая сильное давление на языковую модель для понимания и обработки информации в рамках ограниченного контекстного окна.

Это, вероятно, ухудшает качество принятия решений. Многоагентная система, которая распределяет задачи на основе источника данных и модальности, может повысить производительность.
Несмотря на то, что такие подходы, как STOCKAGENT, используют многоагентную систему для индивидуальных задач торговли акциями. Тем не менее, их процессы принятия решений в значительной степени зависят от обширных обсуждений между многочисленными агентами LLM, что приводит к высоким коммуникационным расходам и увеличению времени обработки. Кроме того, отсутствие четкой цели оптимизации в таких системах может поставить под угрозу эффективность результатов.


В ответ на эти нерешенные вопросы мы предлагаем свою ситему, многоагентную структуру на основе LLM для критически важных задач принятия финансовых решений, таких как торговля ценными бумагами и управление портфелем.
Наш основной вклад таков: 
- Вдохновленные реальными инвестиционными ролями, мы разработали коммуникационную структуру. Эта структура распределяет финансовые данные из различных источников и методов среди соответствующих функциональных аналитиков, что позволяет им сосредоточиться на создании и обоснования основных инвестиционных идей и показателей из одного источника информации.
- Введен контроль рисков со стороны менеджера.
- Добавлена коммуникация между агентами: Кооперирование и сопреничесво.
- Использование MADDPG — это расширение алгоритма DDPG для многоагентных систем. Основная идея заключается в том, что каждый агент имеет свою акторную сеть, но критическая сеть каждого агента учитывает состояния и действия всех агентов.
- Это обобщенная структура, которая может осуществлять не только торговлю ценными бумагами, но и управление портфелем.
- В рамках контроля за риском осуществляется использование квантильной меры риска из количественного финансирования, условного значения риска (CVaR).
- Использщование метода вербального подкрепления.
- Эти инвестиционные убеждения основаны на основных изменениях прибыли и убытков на уровне траектории (PnL)
- Результаты исследования иллюстрируют значительную эффективность нашего дизайна контроля рисков в управлении рыночными рисками и повышении эффективности торговли.

# Связанные работы
**LLM Agents for Financial Decision Making** Существуют значительные усилия, направленные на разработку агента LLM общего назначения для последовательного принятия решений, и такой тип задач часто включает в себя эпизодические взаимодействия с окружающей средой. Кроме того, исследователи начали использовать то, как агенты LLM могут лучше выполнять более сложные задачи по принятию решений из финансов, 
в которых существуют более нестабильные среды, что приводит к тому, что многочисленные непредсказуемые элементы могут затуманить способность агента точно размышлять о причинах плохих результатов принятия решений. FinMem повышает эффективность торговли одними акциями за счет встраивания модулей памяти с агентом LLM для отражения-усовершенствования, а FinAgent улучшил торговую прибыль за счет использования внешнего количественного инструмента для борьбы с нестабильной средой.

**Multi-Agent System and Communication Structures** В традиционных многоагентных системах путь общения агентов заранее определен, например, обмен данными или наблюдения за состоянием. Появление широкоязыковой модели обеспечивает гибкость для понятных для человека коммуникаций, поэтому некоторые работы пытаются повысить способность принимать решения многоагентной системы на основе LLM, позволяя агентам участвовать в дискуссиях или дебатах. Аналогичная стратегия взаимной связи также использовалась многоагентной системой для финансовых
. Однако такой подход не является оптимальным для финансовых задач с единой целью, которые приоритизируют прибыль, потому что они страдают от потенциально неоднозначных целей оптимизации и не могут контролировать ненужные коммуникационные расходы.

**Prompt Optimization and Verbal Reinforcement** Для улучшения рассуждений или принятия решений агентами LLM было предложено множество методов оперативной оптимизации, таких как ReAct, Chain of Thought (CoT), Tree of Thoughts (ToT), ART, предназначенных для того, чтобы агенты LLM могли автоматически генерировать промежуточные этапы рассуждений в виде итеративной программы. 
Кроме того, чтобы заставить агента LLM принимать решения, как люди, и генерировать более понятные тексты для рассуждений, некоторые исследователи рекомендуют включать когнитивные структуры. Вдохновленный этими предыдущими работами и алгоритмами RL, вербальное подкрепление было разработано для агентов LLM таким образом, чтобы они могли обновлять действия на основе итеративной саморефлексии, 
интегрируя дополнительный LLM в качестве оптимизатора.

# Постановка задачи
**Многоагентная система обучения с подкреплением** представляет собой набор агентов, взаимодействующих в общей среде. Каждый агент принимает решения, основываясь на своем состоянии, и получает вознаграждение, которое зависит от его действий и, возможно, действий других агентов.
- **Цель каждого агента:** Максимизировать ожидаемое суммарное вознаграждение, учитывая взаимодействие с другими агентами.
- **Среда:** Фондовый рынок, представленный котировками акций, индексами, макроэкономическими показателями и новостями.
- **Агенты:** Торговые агенты, принимающие решения о покупке или продаже акций.

## Состояние агента
Состояние агента на шаге времени t представлено вектором:

$$
s_{t} = \[p_{t}, i_{t}, m_{t}, n_{t}, b_{t}\]
$$

где:
- $$p_{t}$$ - вектор цен акций на момент t.
- $$i_{t}$$ - значения рыночных индексов на момент t.
- $$m_{t}$$ - макроэкономические показатели на момент t.
- $$b_{t}$$ - нормализованный баланс агента на момент t
- $$n_{t}$$ - векторное представление новостей на момент t, полученное с помощью модели обработки естественного языка (LLM)


### Вектор цен
Вектор цен берется из данных bloomberg. Будут браться топ 50 кампаний по обороту из рейтинга SNP500

### Рыночные индексов (Список можно дополнять)
#### Акционерные индексы (Stock Market Indices)
- **S&P 500:** Включает 500 крупнейших компаний США по рыночной капитализации. Широко используется как основной показатель состояния американского фондового рынка.
- **Russell 3000:** Охватывает 3000 крупнейших компаний США, представляя более широкий спектр рынка по сравнению с S&P 500.
- **Russell 2000:** Фокусируется на 2000 малых компаний США, часто используется для оценки динамики малых капитализаций.
- **NASDAQ Biotechnology Index:** Включает компании из биотехнологического сектора, торгующиеся на NASDAQ.
- **S&P 500 Information Technology:** Отражает производительность технологического сектора среди компаний S&P 500.
- **MSCI Global Financials Index:** Фокусируется на финансовом секторе глобальных рынков.
- **Russell 1000 Growth:** Включает крупные компании с высоким потенциалом роста.
- **Russell 1000 Value:** Содержит крупные компании, оцениваемые как недооценённые по сравнению с их фундаментальными показателями.
- **S&P 500 Dividend Aristocrats:** Включает компании S&P 500, стабильно выплачивающие дивиденды в течение как минимум 25 лет.
#### Международные индексы
- **MSCI EAFE (Europe, Australasia, Far East):** Охватывает развитые рынки Европы, Австралазии и Дальнего Востока, исключая США и Канаду.
- **FTSE 100:** Включает 100 крупнейших компаний, торгующихся на Лондонской фондовой бирже.
- **Nikkei 225:** Отражает состояние японского фондового рынка, включающий 225 ведущих компаний.
#### Облигационные индексы (Bond Market Indices)
- **Bloomberg Barclays US Aggregate Bond Index:** Представляет широкий спектр облигаций США, включая государственные, корпоративные, ипотечные и другие долговые инструменты.
- **ICE BofA High Yield Index:** Фокусируется на высокодоходных корпоративных облигациях (junk bonds).
#### Товарные индексы (Commodity Indices)
- **S&P GSCI (Goldman Sachs Commodity Index):** Один из наиболее известных товарных индексов, охватывающий широкий спектр сырьевых товаров.
- **Bloomberg Commodity Index (BCOM):** Включает различные товары, такие как энергия, металлы, сельскохозяйственные продукты.
#### Индексы волатильности
- **VIX (CBOE Volatility Index):** Известен как "индекс страха", измеряет ожидаемую волатильность рынка на основе опционов S&P 500.
- **VXN:** Аналог VIX, но для NASDAQ 100.
#### Индексы недвижимости (Real Estate Indices)
- **FTSE NAREIT All REITs Index:** Включает все публично торгуемые Real Estate Investment Trusts (REITs) в США.
- **MSCI US REIT Index:** Оценивает производительность американских REITs.
#### Алтернативные индексы
- **ESG индексы** (Environmental, Social, Governance)
- **MSCI KLD 400 Social Index:** Включает компании, демонстрирующие высокие показатели в области экологической, социальной ответственности и управления.
- **S&P 500 ESG Index:** Адаптация S&P 500 с учетом критериев ESG.
#### Инновационные и технологические индексы
- **NASDAQ-100 Technology Sector Index:** Включает технологические компании из NASDAQ-100.
- **ARK Innovation ETF (ARKK):** Хотя это ETF, он отслеживает индекс компаний, ведущих инновации в различных секторах.
#### Индексы по методологии
##### Цена-взвешенные индексы
- **Dow Jones Industrial Average (DJIA):** Взвешен по цене акций, включающий 30 крупных компаний США. Одна из старейших и наиболее известных биржевых индексов.
##### Рыночной капитализации взвешенные индексы
- **S&P 500:** Взвешен по рыночной капитализации, что означает, что компании с большей рыночной стоимостью имеют больший вес в индексе.
- **MSCI World Index:** Включает крупные и средние компании из 23 развитых стран, взвешенные по рыночной капитализации.
##### Равновзвешенные индексы
- **S&P 500 Equal Weight Index:** Каждая компания в индексе имеет равный вес, независимо от её рыночной капитализации.
- **Invesco S&P 500 Equal Weight ETF (RSP):** ETF, отслеживающий равновзвешенный S&P 500.
##### Специализированные индексы
- **S&P 500 PutWrite Index (PUT):** Стратегия, основанная на продаже опционов пут на S&P 500.
- **S&P 500 Low Volatility Index:** Включает акции S&P 500 с низкой волатильностью.
##### Индексы малых капитализаций
- **Russell 2000:** Один из основных индексов для оценки малых капитализаций США.
- **S&P SmallCap 600:** Включает 600 малых компаний США с высокой ликвидностью.

#### Макроэкономические показатели
1. Валовой внутренний продукт (ВВП) и валовый национальный продукт (ВНП)
2. Инфляция
3. Уровень безработицы
4. Процентные ставки
5. Платежный баланс
6. Государственный долг
7. Промышленное производство
8. Индекс потребительских цен (ИПЦ)
9. Индекс деловой активности (PMI)
10. Розничные продажи
11. Производственные заказы
12. Индекс доверия потребителей
13. Объем инвестиций
14. Заработная плата
15. Торговый баланс
16. Валютный курс
17. Демографические показатели
18. Уровень бедности
19. Уровень занятости
20. Сбережения населения
21. Ценовые индексы
22. Бюджетный дефицит
23. Экспорт и импорт
24. Инвестиции в основной капитал
25. Строительство
26. Уровень предпринимательской активности
27. Энергетические показатели

#### Веткорное представление новостей
Используется модель обработки естественного языка (LLM), в данном случае предобученная модель BERT, для преобразования текстовых новостей в векторные представления.
Вектор новостей $$n_{t}$$ получается как среднее скрытых состояний последнего слоя BERT:
$$n_{t} = BERT(news_{t})$$,  где $$news_{t}$$ — текст новостей на момент t.

#### Нормализованный баланс агента на момент t
Агрегированное состояние счета (profit and loss) агента на момет времени t 

## Действие агента
Действие агента на шаге времени t представлено вектором

$$
a_{t} = \[a^{(1)}_{t}, a^{(2)}_{t}, .... , a^{(n)}_{t}\]
$$


$$a^{(i)}$$ - непрерывное значение, представляющее количество акций  i-го типа, которое агент решает купить (>0) или продать (<0).
Пространство действий: Непрерывное, многомерное. Для каждой акции агент выбирает количество для покупки или продажи.

## Вознаграждение (Reward)
Вознаграждение агента на шаге времени t определяется как изменение его общего капитала:


### 2.2. Блок-схема решения

Нет

### 2.3. Этапы решения задачи

### 2.3.1 Выбор метрик

**Оффлайн-метрики** - это метрики, оценивающие производительность системы  на основе исторических данных и фактических результатов. 

![Распределение транзакций в исходной системе](img/Untitled.jpeg)

Распределение транзакций в исходной системе

В соответствии с бизнес метрикой наша основная цель уменьшить убытки от штрафов, что соответствует метрике Recall. А также сохранить пользователей и не создавать лишних блокировок, что соответствует Specificity. На данных исторической системы определили Recall@Specificity 99.85% = 50%  и считаем целью увеличить Recall при условии неухудшения Specificity. Таким образом, мы предполагаем, что эта метрика коррелирует с бизнес метрикой. Соотвественно целевая метрика обучения **Recall@Specificity > 99.85%**. 

Метрика Recall@Specificity является гладкой монотонной, что позволяет использовать быстрый бинарный поиск для определения порога и немного увеличивает скорость последовательности действий обучения.

**Онлайн метрики** -  это метрики, которые мы можем получить во время работы системы. 

Мы не можем иметь какой-либо хорошей онлайн метрики. Но такая метрика понадобится для проведения AB-теста. Поэтому придется использовать прокси-метрики, такие как количество жалоб и подобные.

**Технические метрики** - это метрики, связанные с производительностью и скоростью работы системы антифрод. Некоторые примеры таких метрик включают:

- Скорость обработки транзакций антифрод системой: эта метрика измеряет время, затраченное системой антифрод на обработку каждой транзакции. Ожидается, что время обработки будет ≤ *1c*, чтобы не ухудшать пользовательский опыт и обеспечивать быструю реакцию на мошеннические операции.

Выбор этих метрик позволяет нам оценить эффективность системы  с различных сторон, включая бизнес-показатели, точность обнаружения мошеннической активности и производительность системы. Это позволит нам принять более информированные решения и улучшить антифрод систему  в соответствии с требованиями проекта и бизнес-целями.

### 2.3.2 Определение объекта и таргета

- Объект - транзакция, которая включает уникальные идентификаторы получателя и отправителя, время и другие поля.
- Целевая переменная: мошенническая транзакция помечается как "1", а немошенническая транзакция - как "0".

### 2.3.3 Сбор данных

Целевая переменная, помещающая транзакции как фрод, получаем из трех источников

- помеченные аналитиками транзакции как фрод
- транзакции, которые были помечены rule-based системой как гарантированный фрод и отклоненные сразу (black-list)
- транзакции, по котором пришли штрафы
- транзакции, по которым были подтвержденные жалобы на фрод

Информация о получении данных:

| Название данных | Есть ли данные в компании (если да, название источника/витрины) | Требуемый ресурс для получения данных (какие роли нужны) | Проверено ли качество данных (да, нет) |
| --- | --- | --- | --- |
| Истории транзакций за последние N-месяцев | Да, хранятся в базе данных компании | Product owner | Нет |
| Целевая переменная | Да, содержится в данных истории транзакций | Product owner/ Отдел аналитиков | Нет |
| Время транзакции | Да, содержится в данных истории транзакций | Product owner | Нет |
| Сумма транзакции | Да, содержится в данных истории транзакций | Product owner | Нет |
| ID отправителя | Да, содержится в данных истории транзакций | Product owner | Нет |
| ID получателя | Да, содержится в данных истории транзакций | Product owner | Нет |
| Канал перевода | Да, содержится в данных истории транзакций | Product owner | Нет |
| ID банка отправителя | Да, содержится в данных истории транзакций | Product owner | Нет |
| ID банка получателя | Да, содержится в данных истории транзакций | Product owner | Нет |
| Назначение платежа* | Да, содержится в данных истории транзакций | Роль с доступом к базе данных | Нет |
| Данные клиента (соц демо, последние операции) | Да, хранятся в базе данных компании | Product owner | Нет |
| Данные устройства, с которого переводят (версия ОС, имя приложения и т.п.) | Да, хранятся в базе данных компании | Product owner | Нет |

### 2.3.4 Подготовка данных

Этап 0: анализ временного ряда на в всех доступных данных (3 года). Проверка гипотез о сезонности транзакций в рамках разных регионов. По результату: принятие решение об отрезке времени, на котором будет основано итоговое решение задачи. 

Этап 1: Проверка каждого признака

| Название данных | Требования |
| --- | --- |
| Истории транзакций за последние N-месяцев | - Убедиться, что данные охватывают выбранный период без пропусков. |
| Целевая переменная | - Определить критерии классификации мошеннических операций. |
| Время транзакции | - Убедиться в правильном формате данных времени. |
| Сумма транзакции | - Проверить отсутствие пропусков в колонке с суммой транзакции. |
| ID отправителя | - Проверить отсутствие пропусков в колонке с ID отправителя. |
| ID получателя | - Проверить отсутствие пропусков в колонке с ID получателя. |
| Канал перевода | - Проверить наличие корректных значений каналов перевода. |
| ID банка отправителя | - Проверить наличие корректных значений ID банка отправителя. |
| ID банка получателя | - Проверить наличие корректных значений ID банка получателя. |
| Назначение платежа* | - Проверить наличие и корректность данных в поле назначения платежа. |
| Данные клиента (соц демо, последние операции) | - Определить необходимые атрибуты клиента и проверить их наличие и корректность. |
| Данные устройства, с которого переводят | - Определить необходимые атрибуты устройства и проверить их наличие и корректность. |

Этап 2: Оценка сбалансированности набора данных (сопоставимы ли соотношения между классами в рамках датасета)

**Результат:** Чистый и корректный набор данных для задачи

### 2.3.4 Работа с  признаками

- Проверка распределения транзакций во временном ряду в рамках выбранного отрезка времени, проверка отсутствия значительных пропусков.
- Оценка распределений целевого признака для типа перевода Card2Card/Account2Account. Проверка гипотез о специфичностях категорий (разные по сумме типичные транзакции, разная частота транзакций, разное соотношение мошеннических транзакций);
- Проверка гипотез о сезонности. Уточнение отрезка времени, необходимого для пересчета модели;
- Оценка корректности разметки данных. Часть данных может быть передана на повторную проверочную разметку для подтверждения уверенности в разметке;
- Оценка важности признаков для baseline и MVP, формулирование гипотез для новых признаков с целью улучшения качества системы (возможные варианты признаков — время между транзакциями одного пользователя, обыкновенная частота транзакций для времени/дня недели, неизвестный (новый) девайс пользователя);
- Признак времени с последний транзакции, количества транзакций за последние N минут;
- Признак нового устройства, локации;

**Результат:** 
1. Включение или исключение фактора полной даты транзакции в датасет;
2. Проверено влияние фактора ошибки разметчиков в наборе данных;
3. Сформулированы гипотезы о возможных новых признаках для обогащения набора данных;
4. Сделана оценка оптимального периода перетренировки модели, в том числе с учетом задержки в получении разметки новых данных.

**Необходимая проверка бизнеса:** оценка согласованности гипотез о новых признаках с предметной областью

### 2.3.5 Оценка поведения baseline

### 2.3.5.1 Применение методов интепретации к признакам для baseline

**Возможные методы:** Permutation importance, Individual Conditional Expectation and Partial dependence plots

**Результат:** При необходимости коррекция признаков на основе выводов, извлекаемых с интерпретацией

**Необходимая проверка бизнеса:** оценка согласованности интерпретаций с предметной областью для избежания искажений 

### 2.3.6 Подготовка тренировочного и тестового наборов данных

На основе EDA для формирования train, valid и test set планируется учесть следующие составляющие:

1. Нюансы временного ряда — разбиение датасета без внесения информации о потенциальном будущем в модель;
2. Разделение с одинаковым распределением таргета, так как есть дисбаланс. Допустимо использовать undersampling и oversampling для выравнивания дисбаланса, а при необходимости более сложных методов SMOTE/ADASYN (увеличение малого класса за счёт представителей выпуклых комбинаций пар)

**Результат:** Сформированы выборки для обучения, валидации и финального тестирования модели

### 2.3.7 Валидация

- В качестве валидационной стратегии будет использоваться Time-Series Validation, которая учитывает временные аспекты данных и обеспечивает более реалистичную оценку ее производительности.

![stream_prequential_1block.png](img/stream_prequential_1block.png)

- При использовании Time-Series Validation датасет делится на K фолдов, где каждый фолд представляет собой свой временной промежуток данных, при этом заранее делается отложенная выборка, представляющая последнюю часть данных, на которой делаются финальные замеры качества. Эта отложенная выборка не меняется для всех фолдов. Это означает, что каждый фолд содержит информацию о транзакциях, произошедших в определенный период времени.
- Кроме того, между тренировочным и валидационным набором данных также существует временной отрезок (скорее всего это 1 или 2 недели). Это означает, что модель обучается на данных, предшествующих временному отрезку, и затем тестируется на данных, находящихся в этом отрезке. Таким образом, мы имитируем реальные условия, в которых модель должна делать прогнозы на будущих данных, основываясь только на прошлых данных.
- Использование Time-Series Validation позволяет учесть хронологический порядок данных и оценить производительность модели в реалистичных условиях.

### 2.3.8 Выбор архитектуры модели для решения задачи

### От baseline к полноценной модели

В качестве отправного качества планируется взять качество нынешней rule-based системы.

**Особенности работы**

Учитывая особенности исторической системы, оставляем возможность, что модель имеет выход Разрешить/Сомнительно/Запретить. Сомнительные отправляются на проверку аналитиками.

Система, как было указано выше, будет работать параллельно с rule-based моделью. Rule-based модель остается работать параллельно, а конечный вердикт выносится на основе предикта от обеих систем. Rule-based модель может включать хорошо детектируемые случаи, черные списки, а так же направлять не дополнительную проверку транзакции с большими суммами.

Заявки для аналитиков обрабатываются очередью с учетом приоритета, который определяется на основе score и суммы транзакции.

В случае небольшой загруженности аналитиков, на них назначается случайная часть транзакций с низким приоритетом для разметки будущего переобучения.

**Этап 1 (MVP)**. Тестирование избранных алгоритмов — `ONE-CLASS SVM`, `Logistic Regression`, `Random Forest`, `XGBoost`. Данные методы способны обнаруживать аномалии в данных, что является важным для предотвращения мошенничества. Все перечисленные модели интерпретируются. По ретроспективе, по сравнению с Rule-Based подходом, такие методы способны, при корректном обучении, кратно улучшить качество системы. После этого следует этап 2.3.10 и итерационная доработка. 

**Этап 2**. Тестирование оптимизаций для алгоритмов, бьющих baseline.

Данный этап включает в себя варьирование гиперпараметров — математических и эвристических показателей модели, для её улучшения, а так же подбор подходящих признаков

**Этап 3**. Разработка архитектуры нейронной сети для решения задачи - `Autoencoder`, `Autoencoder+GAN`. Плюсы нейросетевего подхода является предполагаемая более высокая точность, минусами — более сложная интерпретируемость с использованием соответствующих библиотек.   

**Autoencoder** обучается на на обычных транзакциях, ошибка восстановления является критерием для определения как фрод

**Autoencoder+GAN** основан на следующем [принципе](https://arxiv.org/pdf/1908.11553.pdf):

1. Берем хорошие транзакции, учим на них Autoencoder
2. Репрезентация после encoder, полученная от sparse autoencoder становится новыми фичами
3. Учим GAN, где реальные образцы - это репрезентация существующих хороших транзакций с пункта 2, а фейковые шум
4. Снимаем с GAN дискриминатор и используем его теперь чтобы принять решение о транзакции (прогнанной сначала через Autoencoder) - фрод или нет

**Этап 4**. Сопоставление качества проверенных моделей по целевой метрике. Выбор конечной архитектуры/структуры модели.  

1. Берем хорошие транзакции, учим на них sparse autoencoder
2. Репрезентация (code после encoder) полученная от sparse autoencoder становится новыми фичами
3. Учим GAN, где реальные образцы - это репрезентация существующих хороших транзакций с пункта 2, а фейковые, понятное дело, шум
4. Снимаем с GAN дискриминатор и используем его теперь чтобы принять решение о транзакции (прогнанной сначала через sparse autoencoder) - фрод или нет
5. Берем хорошие транзакции, учим на них sparse autoencoder
6. Репрезентация (code после encoder) полученная от sparse autoencoder становится новыми фичами
7. Учим GAN, где реальные образцы - это репрезентация существующих хороших транзакций с пункта 2, а фейковые, понятное дело, шум
8. Снимаем с GAN дискриминатор и используем его теперь чтобы принять решение о транзакции (прогнанной сначала через sparse autoencoder) - фрод или нет

**Результат:** Выбрано три приоритетных типа модели для дальнейшей оптимизации

### 2.3.9 Методы интерпретации

- Используем анализ интерпретируемости по всей модели для выявления паттернов и значимых признаков
- Используем анализ интерпретируемости отдельного предикта для обработки ошибок и анализа конкретных видов фрода
- Способы интерпретации моделей**:**
    - Для стандартных методов Isolation Forest, One-class SVM, LR, RF, XGBoost существуют стандартные интерпретации, например на SHAP
    - Для нейросетевых моделей можем так же использовать SHAP. Например [тут](https://heka-ai.medium.com/detection-and-interpretation-of-outliers-thanks-to-autoencoder-and-shap-values-c8bcdc5ebc1e) есть применение SHAP для интерпретации Autoencoder. Для связки Autoencoder+GAN интерпретация может быть сложнее, но принцип останется тот же, так как в режиме инференса модель будет представлять единую нейросеть.
    - Так же в качестве возможных методов интерпретации, которые можно подсчитать почти для любого типа моделей — Permutation importance, Individual Conditional Expectation, Counterfactual Explanations and Partial dependence plots.

### 2.3.9 Анализ ошибок

Анализ наличия и природы ошибок в MVP. Проверка модели при помощи методов интерпретации, оценка корректности гиперпараметров. При необходимости возвращение к пунктам 2.3.4—2.3.8.

**Результат:** Модель, достигающая качества на тестовом наборе данных, соответствующего бизнес требованиям. 

**Необходимая проверка бизнеса:** Согласованность качества модели с требованиями бизнеса;

## 3. Подготовка пилота


### 3.1. Способ оценки пилота

В исторической системе транзакции поступали на RuleBased блок. Добавляем  Switch, который будет разделять и/или перенаправлять транзакции на новую или историческую системы. Разделение транзакций позволит сделать AB-тест, посчитать офлайн метрики и оценить эффективность решения.

Для AB-теста мы не имеем хорошую онлайн метрику, поэтому можем оценивать прокси метрики, такие как жалобы от пользователей. Мы разбиваем транзакции на группы AB или AAB. Так же проводим отложенную оценку офлайн и бизнес метрик, возможно в несколько временных точках, для сравнения.

Последовательность подготовки AB-теста:

- Определить метрики:
    - В моменте сразу после проведения АБ используем количествово жалоб, но спустя время (1-2 месяца, как дойдет разметка) еще раз смотрим и целевую бизрес метрику. 
    - Вспомогательные метрики, которые должны показать, что тест идет так как предполагалось и не затронул другие области, например это количество транзакций, сумма транзакций и т.д.
    - Контрольные метрики, которые будут позволять отлавливать проблемы, такие как время проведения транзакции, обращения в поддержку и другие
- Определить с бизнесом, какой эффект мы собираемся детектировать. Определить уровень значимости и допустимый уровень ошибок второго рода
- Отбор групп должен производиться по пользователям, а не по транзакциям, чтобы избежать зависимости данных. Обычно, в организации проводятся несколько AB-тестов и существует система разбиения с одинарным или двойным хешированием
- Оценить возможности увеличения мощности теста, такие как уменьшение дисперсии, стратификация, CUPED и другие
- Оценив требуемый минимальный детектируемый эффект и сезонность данных определить требуемый процент транзакций и длительность проведения теста. Если в пределах проведения теста есть волатильность целевой метрики, то нужно использовать нестандартные тесты, которые допускают волатильность метрики
- Подготовить техническую основу теста
- Проверить тест на исторических данных и подтвердить уровни ошибок первого и второго рода

Во время проведения пилота

- При первом запуске предусмотреть “параллельный” запуск, при котором решение, принятое системой по каждой транзакции, не идет в работу, а сохраняется отдельно для анализа
- Возможно, использовать постепенное развертывание, при котором мы запускаем на новую систему не сразу на всех планируемых транзакциях
- Следить за мониторингом корректности проведения эксперимента

### 3.2. Что считаем успешным пилотом

По завершению теста принимаем решение. 

Успешным проведение пилота считается при условиях:

- Целевая метрика статистически значимо выросла
- Контрольные метрики не упали
- Вспомогательные метрики не противоречат друг другу и целевой

В случае успеха пилота переходим к внедрению. В случае неуспеха определяем и исправляем ошибки или улучшаем модель и систему, либо переходим к другим типам решения или отказываемся от решения.

При любом исходе пилота мы собираем аналитику, новые идеи и гипотезы. А так же сохраняем все результаты.

Для измерения критерия успеха используем окупаемость системы. Доходом является уменьшение штрафов и негативного влияния. Затратами являются затраты на разработку и сопровождение. При окупаемости за N месяцев считаем успешным внедрением. 

1. Целевая метрика статзначимо выросла более чем на Х% при условии отсутствия падения контрольных метрик<br>
2. Снижение количества пропущенных фродовых операций и соответствующее уменьшение размера штрафов от регулятора.<br>
3. Уменьшение количества неоправданно заблокированных транзакций для улучшения клиентского опыта.<br>
4. Увеличение эффективности работы аналитиков антифрод системы за счет снижения количества требующих ручной проверки транзакций.<br>
5. Поддержание или увеличение клиентской базы за счет улучшения качества антифродовых мер и соответствующего увеличения доверия.</span> 

## 4. Внедрение

### 4.1. Архитектура решения

Архитектура решения. 

**Белые блоки** - историческая система

**Цветные блоки** - новая система

**Темный цвет** выделяет блоки, относящиеся к пилоту

![HLD.png](img/HLD.png)



### 4.2. Описание инфраструктуры и масштабируемости

- Несмотря на сравнительно небольшую среднюю нагрузку в 1.5 транзакций в секунду, нужно предусмотреть отработку пиковых нагрузок, связанных с цикличностью нагрузки и фрод атаками
- Очередь сообщений может быть организована на брокере сообщений, например Kafka
- Инференс модели может быть организован в Kubernetes кластере с автоматическим подключением нод на основе события системы мониторинга

### 4.3. Мониторинг

Сервис мониторинга контролирует:

- Технические метрики
- Нагрузка на сервер
- Нагрузка на сервис
- Прокси метрика количество жалоб
- Офлайн метрики
- Дата дрифт

В результате превышения порогов вызываются 

- переключение на предыдущую или историческую модель
- оповещения аналитикам и другим службам
- запуск пайплана переобучения

### 4.4. Требования к работе системы

- Пропускная способность и задержка обеспечивается структурой и возможностью масштабирования

### 4.5. Безопасность системы

Обеспечивается отсутствием внешних связей

### 4.6. Безопасность данных

Безопасность данных обеспечивается 100% хранением на внутренних серверах

### 4.7. Издержки

- Расчетные издержки на работу системы в месяц
- Зарплаты 3 МЛ специалистов на период от 2 - 4 месяцев для создания MVP
- Затраты на инфраструктуру

### 4.8. Риски

- Риск пропустить фродовую операцию с большой суммой - создание правил проверки аналитиками всех операций с большой суммой, подкрепляя прогнозом и интерпритацией кейса, полученными моделью
- Сбой работы модели в результате хакерских атак - создание standby сервисов на уровне инфраструктуры;
- Технический сбой в модели - переключаемся на предыдущую или историческую
- Сбой метрик мониторинга выше среднего порога - оповещение
- Сбой метрик мониторинга выше верхнего порога - переключаемся на предыдущую или историческую
- Риск пропуска данных - контролируется на уровне целостности данных
- Большое количество перенаправляется на аналитиков - создание алертов для подключения аналитиков, временное оперативное изменение порогов модели
- Новый тип фрода - быстрое добавление новых правил, разметка, переобучение, проверка