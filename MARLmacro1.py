import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Инициализация модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Определение среды
class EconomyEnv(gym.Env):
    def __init__(self):
        super(EconomyEnv, self).__init__()
        
        # Параметры экономики
        self.num_households = 5
        self.num_firms = 3
        
        # Пространства действий и состояний
        # Домохозяйства: [трудовое предложение]
        self.action_space_household = spaces.Box(low=0, high=1, shape=(self.num_households,), dtype=np.float32)
        
        # Фирмы: [заработная плата, цена товара, инвестиции]
        self.action_space_firm = spaces.Box(low=0, high=10, shape=(self.num_firms, 3), dtype=np.float32)
        
        # Правительство: [налоговая ставка на доходы, налоговая ставка на прибыль фирм, процентная ставка]
        self.action_space_government = spaces.Box(low=0, high=0.5, shape=(3,), dtype=np.float32)
        
        # Состояния
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_households * 3 + self.num_firms * 4 + 4,), dtype=np.float32)
        
        # Логи для рассуждений и общения
        self.reasoning_logs = []
        self.communication_logs = []
        
        self.reset()
    
    def reset(self):
        # Сброс состояния среды
        self.households = [{
            'wealth': 100.0,
            'labor_supply': 0.0,
            'consumption': 0.0,
            'inflation_expectation': 0.02,  # Ожидание инфляции
            'message': ''  # Сообщение для общения
        } for _ in range(self.num_households)]
        self.firms = [{
            'capital': 100.0,
            'price': 1.0,
            'wage': 1.0,
            'investment': 0.0,
            'production': 0.0,
            'labor_demand': 0.0,
            'technology_level': 1.0,  # Уровень технологий
            'message': ''  # Сообщение для общения
        } for _ in range(self.num_firms)]
        self.government = {
            'tax_rate_income': 0.2,
            'tax_rate_corporate': 0.2,
            'interest_rate': 0.05,
            'budget': 0.0,
            'inflation_rate': 0.02,  # Текущая инфляция
            'message': ''  # Сообщение для общения
        }
        self.market = {
            'total_labor_supply': 0.0,
            'total_labor_demand': 0.0,
            'total_goods_supply': 0.0,
            'total_goods_demand': 0.0,
            'exchange_rate': 1.0  # Курс валюты для внешнего сектора
        }
        self.external_sector = {
            'foreign_demand': 50.0,  # Спрос со стороны внешнего рынка
            'foreign_supply': 50.0   # Предложение со стороны внешнего рынка
        }
        self.time_step = 0
        
        # Очистка логов
        self.reasoning_logs.clear()
        self.communication_logs.clear()
        
        return self._get_obs()
    
    def step(self, actions):
        # Разделяем действия агентов
        household_actions = actions['households']  # Предложение труда от домохозяйств
        firm_actions = actions['firms']            # [заработная плата, цена товара, инвестиции] от фирм
        government_action = actions['government']  # [налоговая ставка на доходы, налоговая ставка на прибыль, процентная ставка]
        messages = actions.get('messages', {})     # Сообщения от агентов
        
        # Логирование сообщений
        self._log_communications(messages)
        
        # Обновляем сообщения агентов
        for i, household in enumerate(self.households):
            household['message'] = messages.get(f'household_{i}', '')
        for i, firm in enumerate(self.firms):
            firm['message'] = messages.get(f'firm_{i}', '')
        self.government['message'] = messages.get('government', '')
        
        # Обновляем действия правительства
        self.government['tax_rate_income'], self.government['tax_rate_corporate'], self.government['interest_rate'] = government_action
        
        # Информационная асимметрия и иррациональное поведение
        info_asymmetry = np.random.choice([True, False], p=[0.3, 0.7])  # 30% агентов имеют ограниченную информацию
        irrational_behavior = np.random.choice([True, False], p=[0.2, 0.8])  # 20% агентов ведут себя иррационально
        
        # Домохозяйства предлагают труд
        for i, household in enumerate(self.households):
            labor_supply = household_actions[i]
            reasoning = f"Household {i} decides labor supply {labor_supply:.2f}."
            if info_asymmetry:
                labor_supply *= np.random.uniform(0.8, 1.2)
                reasoning += " Due to limited information, adjusts labor supply."
            if irrational_behavior:
                labor_supply *= np.random.uniform(0.5, 1.5)
                reasoning += " Shows irrational behavior, further adjusts labor supply."
            labor_supply = np.clip(labor_supply, 0, 1)
            household['labor_supply'] = labor_supply
            self.market['total_labor_supply'] += labor_supply
            self._log_reasoning(reasoning)
        
        # Фирмы определяют спрос на труд, заработную плату и цену товара
        for i, firm in enumerate(self.firms):
            wage, price, investment = firm_actions[i]
            reasoning = f"Firm {i} sets wage {wage:.2f}, price {price:.2f}, investment {investment:.2f}."
            if info_asymmetry:
                price *= np.random.uniform(0.8, 1.2)
                reasoning += " Due to limited information, adjusts price."
            if irrational_behavior:
                investment *= np.random.uniform(0.5, 1.5)
                reasoning += " Shows irrational behavior, adjusts investment."
            wage = np.clip(wage, 0, 10)
            price = np.clip(price, 0, 10)
            investment = np.clip(investment, 0, firm['capital'] * 0.2)
            firm['wage'] = wage
            firm['price'] = price
            firm['investment'] = investment
            # Спрос на труд определяется на основе сложной производственной функции
            labor_demand = self._labor_demand(firm)
            firm['labor_demand'] = labor_demand
            self.market['total_labor_demand'] += labor_demand
            self._log_reasoning(reasoning)
        
        # Рыночный механизм для определения равновесной заработной платы
        self._labor_market_clearing()
        
        # Домохозяйства получают заработную плату и потребляют товары
        total_goods_demand = 0.0
        for i, household in enumerate(self.households):
            # Доход после налогов
            income = household['labor_supply'] * self.market['equilibrium_wage']
            net_income = income * (1 - self.government['tax_rate_income'])
            # Обновление богатства с учетом процентной ставки и инфляции
            household['wealth'] = (household['wealth'] * (1 + self.government['interest_rate']) - household['wealth'] * self.government['inflation_rate']) + net_income
            # Ожидание инфляции влияет на потребление
            consumption = self._consumption_decision(household)
            household['consumption'] = consumption
            total_goods_demand += consumption
            self.market['total_goods_demand'] += consumption
            # Уплата налогов
            self.government['budget'] += income * self.government['tax_rate_income']
            reasoning = f"Household {i} earns income {income:.2f}, consumes {consumption:.2f}."
            self._log_reasoning(reasoning)
        
        # Внешний сектор влияет на спрос
        total_goods_demand += self.external_sector['foreign_demand'] * self.market['exchange_rate']
        self.market['total_goods_demand'] += self.external_sector['foreign_demand'] * self.market['exchange_rate']
        
        # Фирмы производят товары и получают прибыль
        total_goods_supply = 0.0
        for i, firm in enumerate(self.firms):
            production = self._production_function(firm)
            firm['production'] = production
            total_goods_supply += production
            self.market['total_goods_supply'] += production
            # Выручка с учетом экспорта
            revenue = firm['price'] * production + firm['price'] * self.external_sector['foreign_demand'] * self.market['exchange_rate'] / self.num_firms
            # Затраты
            costs = firm['wage'] * firm['labor_demand'] + firm['investment']
            # Прибыль до налогов
            profit = revenue - costs
            # Уплата налогов на прибыль
            net_profit = profit * (1 - self.government['tax_rate_corporate'])
            firm['capital'] += net_profit - firm['capital'] * self.government['inflation_rate']
            # Уплата налогов
            self.government['budget'] += profit * self.government['tax_rate_corporate']
            reasoning = f"Firm {i} produces {production:.2f}, earns profit {net_profit:.2f}."
            self._log_reasoning(reasoning)
        
        # Внешний сектор влияет на предложение
        total_goods_supply += self.external_sector['foreign_supply'] * self.market['exchange_rate']
        self.market['total_goods_supply'] += self.external_sector['foreign_supply'] * self.market['exchange_rate']
        
        # Рыночный механизм для определения равновесной цены
        self._goods_market_clearing()
        
        # Обновление инфляции
        self._update_inflation()
        
        # Обновление финансового сектора
        self._update_financial_sector()
        
        # Вычисление вознаграждений
        rewards = {
            'households': [self._calculate_utility(h) for h in self.households],
            'firms': [f['production'] for f in self.firms],
            'government': self.government['budget']
        }
        
        # Обновляем состояние среды
        obs = self._get_obs()
        self.time_step += 1
        done = self.time_step >= 20  # Эпизод длится 20 шагов
        info = {}
        return obs, rewards, done, info
    
    def _log_reasoning(self, reasoning):
        # Логирование рассуждений
        self.reasoning_logs.append(reasoning)
    
    def _log_communications(self, messages):
        # Логирование сообщений
        for agent_id, message in messages.items():
            log_entry = f"{agent_id} says: {message}"
            self.communication_logs.append(log_entry)
    
    def _get_obs(self):
        # Формируем наблюдение, включающее состояния всех агентов и рынка
        obs = []
        for household in self.households:
            obs.extend([
                household['wealth'],
                household['labor_supply'],
                household['inflation_expectation']
            ])
        for firm in self.firms:
            obs.extend([
                firm['capital'],
                firm['price'],
                firm['wage'],
                firm['technology_level']
            ])
        obs.extend([
            self.government['tax_rate_income'],
            self.government['tax_rate_corporate'],
            self.government['interest_rate'],
            self.government['inflation_rate']
        ])
        return np.array(obs, dtype=np.float32)
    
     def _labor_demand(self, firm):
        # Сложная функция спроса на труд фирмой
        alpha = 0.3  # Эластичность капитала
        beta = 0.7   # Эластичность труда
        A = firm['technology_level']  # Технологический коэффициент
        # Предельный продукт труда с учетом технологического уровня и заработной платы
        marginal_product_labor = beta * A * (firm['capital'] ** alpha) * (firm['labor_demand'] ** (beta - 1))
        # Фирма нанимает труд до тех пор, пока предельный продукт труда равен заработной плате
        labor_demand = ((beta * A * (firm['capital'] ** alpha)) / firm['wage']) ** (1 / (1 - beta))
        return labor_demand
    
    def _labor_market_clearing(self):
        # Определение равновесной заработной платы на рынке труда с учетом инфляции
        total_labor_supply = sum([h['labor_supply'] for h in self.households])
        total_labor_demand = sum([f['labor_demand'] for f in self.firms])
        # Равновесная заработная плата корректируется на инфляцию
        if total_labor_supply > 0:
            wage = (sum([f['wage'] for f in self.firms]) / self.num_firms) * (1 + self.government['inflation_rate'])
            self.market['equilibrium_wage'] = wage
        else:
            self.market['equilibrium_wage'] = 1.0 * (1 + self.government['inflation_rate'])
    
    def _goods_market_clearing(self):
        # Определение равновесной цены на рынке товаров с учетом инфляции
        total_goods_supply = self.market['total_goods_supply']
        total_goods_demand = self.market['total_goods_demand']
        # Равновесная цена корректируется на инфляцию
        if total_goods_supply > 0:
            price = (sum([f['price'] for f in self.firms]) / self.num_firms) * (1 + self.government['inflation_rate'])
            self.market['equilibrium_price'] = price
        else:
            self.market['equilibrium_price'] = 1.0 * (1 + self.government['inflation_rate'])
    
    def _production_function(self, firm):
        # Сложная производственная функция фирмы с технологическим прогрессом
        alpha = 0.3  # Эластичность капитала
        beta = 0.7   # Эластичность труда
        A = firm['technology_level']  # Технологический коэффициент
        labor = firm['labor_demand']
        capital = firm['capital']
        production = A * (capital ** alpha) * (labor ** beta)
        # Технологический прогресс со временем
        firm['technology_level'] *= 1.01
        return production
    
    def _consumption_decision(self, household):
        # Решение о потреблении с учетом ожидания инфляции и иррационального поведения
        consumption = household['wealth'] * 0.5  # Базовое потребление
        # Корректировка на ожидание инфляции
        consumption *= (1 + household['inflation_expectation'])
        # Иррациональное поведение: импульсивные покупки
        if np.random.rand() < 0.1:
            consumption *= np.random.uniform(1.1, 1.5)
        return consumption
    
    def _calculate_utility(self, household):
        # Функция полезности домохозяйства с учетом иррациональности
        consumption = household['consumption']
        leisure = 1 - household['labor_supply']
        eta = 0.5  # Параметр полезности потребления
        theta = 0.5  # Параметр полезности досуга
        # Иррациональное предпочтение потребления
        if np.random.rand() < 0.1:
            eta *= np.random.uniform(1.1, 1.5)
        utility = eta * np.log(consumption + 1e-6) + theta * np.log(leisure + 1e-6)
        return utility
    
    def _update_inflation(self):
        # Обновление инфляции на основе изменения общего уровня цен
        previous_price_level = self.government['inflation_rate']
        current_price_level = self.market['equilibrium_price']
        inflation_rate = (current_price_level - previous_price_level) / previous_price_level
        self.government['inflation_rate'] = inflation_rate
    
    def _update_financial_sector(self):
        # Обновление финансового сектора: начисление процентов и учет инфляции
        for household in self.households:
            household['wealth'] *= (1 + self.government['interest_rate'] - self.government['inflation_rate'])
            # Обновление ожидания инфляции
            household['inflation_expectation'] = self.government['inflation_rate']
    
    def render(self, mode='human'):
        # Вывод текущего состояния экономики
        print(f"Time Step: {self.time_step}")
        print(f"Government Budget: {self.government['budget']:.2f}, Inflation Rate: {self.government['inflation_rate']:.4f}")
        for i, household in enumerate(self.households):
            print(f"Household {i}: Wealth = {household['wealth']:.2f}, Labor Supply = {household['labor_supply']:.2f}, Consumption = {household['consumption']:.2f}")
        for i, firm in enumerate(self.firms):
            print(f"Firm {i}: Capital = {firm['capital']:.2f}, Wage = {firm['wage']:.2f}, Price = {firm['price']:.2f}, Production = {firm['production']:.2f}, Technology Level = {firm['technology_level']:.2f}")
        print("-" * 50)
        # Вывод логов рассуждений и общения
        print("Reasoning Logs:")
        for log in self.reasoning_logs:
            print(log)
        print("Communication Logs:")
        for log in self.communication_logs:
            print(log)
        print("=" * 50)
        # Очистка логов после вывода
        self.reasoning_logs.clear()
        self.communication_logs.clear()

# Функции для генерации начальных стратегий с помощью LLM
def generate_household_action(household_state):
    input_text = f"Домохозяйство с богатством {household_state['wealth']:.2f}, инфляционным ожиданием {household_state['inflation_expectation']:.2f} решает предложить труд."
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=50, do_sample=True)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Простое извлечение числового значения из сгенерированного текста
    labor_supply = np.random.uniform(0, 1)
    # Создание сообщения
    message = f"Я ожидаю инфляцию {household_state['inflation_expectation']:.2f} и предлагаю труд {labor_supply:.2f}."
    return labor_supply, message

def generate_firm_action(firm_state):
    input_text = f"Фирма с капиталом {firm_state['capital']:.2f} и уровнем технологий {firm_state['technology_level']:.2f} устанавливает заработную плату и цену товара."
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=50, do_sample=True)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Простое извлечение числовых значений из сгенерированного текста
    wage = np.random.uniform(0, 10)
    price = np.random.uniform(0, 10)
    investment = np.random.uniform(0, firm_state['capital'] * 0.2)  # Инвестирует до 20% капитала
    # Создание сообщения
    message = f"Я устанавливаю цену {price:.2f} и заработную плату {wage:.2f}."
    return np.array([wage, price, investment]), message

def generate_government_action(government_state):
    input_text = f"Правительство с бюджетом {government_state['budget']:.2f} и инфляцией {government_state['inflation_rate']:.4f} устанавливает налоговые ставки и процентную ставку."
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=50, do_sample=True)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Простое извлечение числовых значений из сгенерированного текста
    tax_rate_income = np.random.uniform(0, 0.5)
    tax_rate_corporate = np.random.uniform(0, 0.5)
    interest_rate = np.random.uniform(0, 0.1)
    # Создание сообщения
    message = f"Я устанавливаю налог на доходы {tax_rate_income:.2f} и процентную ставку {interest_rate:.2f}."
    return np.array([tax_rate_income, tax_rate_corporate, interest_rate]), message

# Инициализация среды
env = EconomyEnv()

# Обучение агентов
# Создаем отдельные модели для домохозяйств, фирм и правительства

# Обучение домохозяйств
class HouseholdPolicy:
    def predict(self, observation, state=None, mask=None, deterministic=False):
        actions = []
        messages = {}
        for i in range(env.num_households):
            idx = i * 3
            household_state = {
                'wealth': observation[idx],
                'labor_supply': observation[idx + 1],
                'inflation_expectation': observation[idx + 2]
            }
            action, message = generate_household_action(household_state)
            actions.append(action)
            messages[f'household_{i}'] = message
        return np.array(actions), messages

# Обучение фирм
class FirmPolicy:
    def predict(self, observation, state=None, mask=None, deterministic=False):
        actions = []
        messages = {}
        offset = env.num_households * 3
        for i in range(env.num_firms):
            idx = offset + i * 4
            firm_state = {
                'capital': observation[idx],
                'price': observation[idx + 1],
                'wage': observation[idx + 2],
                'technology_level': observation[idx + 3]
            }
            action, message = generate_firm_action(firm_state)
            actions.append(action)
            messages[f'firm_{i}'] = message
        return np.array(actions), messages

# Обучение правительства
class GovernmentPolicy:
    def predict(self, observation, state=None, mask=None, deterministic=False):
        offset = env.num_households * 3 + env.num_firms * 4
        government_state = {
            'tax_rate_income': observation[offset],
            'tax_rate_corporate': observation[offset + 1],
            'interest_rate': observation[offset + 2],
            'inflation_rate': observation[offset + 3],
            'budget': env.government['budget']
        }
        action, message = generate_government_action(government_state)
        return action, message

household_policy = HouseholdPolicy()
firm_policy = FirmPolicy()
government_policy = GovernmentPolicy()

# Симуляция
obs = env.reset()
for _ in range(20):
    # Генерируем действия для домохозяйств, фирм и правительства
    household_actions, household_messages = household_policy.predict(obs)
    firm_actions, firm_messages = firm_policy.predict(obs)
    government_action, government_message = government_policy.predict(obs)
    action = {
        'households': household_actions,
        'firms': firm_actions,
        'government': government_action,
        'messages': {**household_messages, **firm_messages, 'government': government_message}
    }
    
    # Шаг симуляции
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break





