# from yahoo_finance import Share
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import yfinance as yf

tf.compat.v1.disable_eager_execution()

def get_price(symbol:str, start:str, end:str):
    stock_price = yf.download(symbol, start=start, end=end)
    return stock_price['Open']

def plot_prices(prices):
    plt.title('Open price')
    plt.xlabel('day')
    plt.ylabel('$ per share')
    plt.plot(prices)
    plt.show()


class DecisionPolicy:
    def select_action(self, current_state):
        pass
    def update_q(self, state, action, reward, next_state):
        pass

# test random decions and try figure out how good random decision works
class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) -1)]
        return action



class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, action, input_dim):
        self.epsilon = 0.9
        self.gamma = 0.001
        self.actions = actions
        output_dim = len(actions)
        h1_dim = 200

        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim])
        self.y = tf.compat.v1.placeholder(tf.float32, [output_dim])

        W1 = tf.Variable(tf.random.normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random.normal([h1_dim, output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.compat.v1.train.AdagradOptimizer(0.01).minimize(loss)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def select_action(self, current_state, step):
        threshhold = min(self.epsilon, step / 1000)
        if random.random() < threshhold:
            #достижение лучшего выбора за счёт вероятности epsilon
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]
        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})

def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist):
    budget = initial_budget
    num_stocks = initial_num_stocks
    share_value = 0
    transitions = list()
    for i in range(len(prices) - hist - 1):
        if i % 100 == 0:
            print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist -1)))
        current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))
        current_portfolio = budget + num_stocks * share_value
        action = policy.select_action(current_state, i)
        share_value = float(prices[i+hist+1])
        if action == 'Buy' and budget >= share_value:
            budget -= share_value
            num_stocks += 1
        elif action == 'Sell' and num_stocks > 0:
            budget +=share_value
            num_stocks -= 1
        else:
            action = 'Hold'
        # print('everything works')
        new_portfolio = budget + num_stocks * share_value
        reward = new_portfolio - current_portfolio
        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))
        transitions.append((current_state, action, reward, next_state))
        # print('work3')
        policy.update_q(current_state, action, reward, next_state)

    portfolio = budget + num_stocks*share_value
    print(portfolio)
    print(action)
    print(current_state)
    return portfolio, action

def run_simulations(policy, budget, num_stocks, prices, hist):
    num_tries = 1
    final_portfolios = list()
    policy_l = []
    for i in range(num_tries):
        final_portfolio, action = run_simulation(policy, budget, num_stocks, prices, hist)
        final_portfolios.append(final_portfolio)
        policy_l.append(action)
        print('Final portf: ${}'.format(final_portfolio))
    plt.title('Fin Portf Value')
    plt.xlabel('Sim #')
    plt.ylabel('Net Worth')
    plt.plot(final_portfolios)
    plt.show()
    return final_portfolios, policy_l




if __name__ == '__main__':
    prices = get_price('MSFT', '1992-07-22', '2016-07-22')
    # plot_prices(prices)
    actions = ['Buy', 'Sell', 'Hold']
    hist = 200
    policy = QLearningDecisionPolicy(actions, hist+2) #RandomDecisionPolicy(actions)
    budget = 100000.0
    num_stocks = 0
    sim_portfolio_value, policy_exec = run_simulations(policy, budget, num_stocks, prices, hist)











