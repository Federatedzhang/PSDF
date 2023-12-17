from statistics import mean, median, stdev
import re

class ContractFeature(object):#定义合约特征类
    def __init__(self, contract_address, ponzi_or_nonponzi):#定义类的相关属性
        self.contract_address = contract_address
        self.ponzi_or_nonponzi = ponzi_or_nonponzi
        self.file_name = self.contract_address + ".csv"

        # features from transactions，提取交易特征
        self.return_rio = 0.0  # known rate：收款人在付款前已投资的比例
        self.A_bal = 0.0  # balance：智能合约的余额
        self.investments_num = 0 # number of investment：投资数
        self.payments_num = 0  # number of payment：支付额数
        self.D_ind = 0  # difference between counts of payment and investment：所有投资者的投资和收益的差异
        self.Pr = 0.0 # the proportion of investors who received at least one payment：至少收到一笔付款的投资者比例
        self.maxpay = 0.0 # the maximum of counts of payments to participants：支付给参与者的最大金额

        # features from opcode，提取操作码特征
        self.action_freq = {}#行为频率
        self.action_ratio = {}#行为比率

        # local variables：局部变量，本地变量，创建相应属性的字典
        self.investors = {}  # format: {address: first_invest_timestamp}
        self.receivers = {}  # format: {address: first_payment timestamp}
        self.payment_counts = {}  # format: {address: number_of_payment_count_to_this_address}
        self.investment_count = {}  # format: {address: number_of_investment_from_this_address}

        self.get_kr_ninv_npay_pr_nmax()#调用get_returnrio_ninv_npay_pr_nmax函数得到相应的特征值
        self.get_balance()

        self.get_d_ind()  # TODO
        self.get_action_frequency()  # get the frequency of each action in the opcode of a smart contract

    def get_kr_ninv_npay_pr_nmax(self):
        with open('./data/transactions/' + self.ponzi_or_nonponzi + '/' + self.file_name, 'r') as tx_file:#self.file_name是/data/transactions/contract_address + ".csv"
            counter = 0
            for tx in tx_file:#循环读取contract_address + ".csv"文件中的内容
                if counter > 0:
                    fields = tx.split(',')
                    from_address = fields[6]
                    # print(from_address)
                    invest_timestamp = fields[2]
                    # print(invest_timestamp)
                    if from_address not in self.investors:#如果from_address不在investors字典中
                        self.investors[from_address] = invest_timestamp#investors[]表示列表，用逗号分隔里面的元素
                        # print(self.investors)
                    else:
                        if invest_timestamp < self.investors[from_address]:
                            self.investors[from_address] = invest_timestamp

                    if from_address not in self.investment_count:
                        self.investment_count[from_address] = 1
                    else:
                        self.investment_count[from_address] += 1
                counter += 1
            self.investments_num = counter - 1#投资数
            # print(self.investors)
            # print('investments_num is {%d}' % self.investments_num)

        with open('./data/internal_transactions/' + self.ponzi_or_nonponzi + '/' + self.file_name, 'r') as in_tx_file:#internal_transactions内部交易
            counter = 0
            for in_tx in in_tx_file:
                if counter > 0:
                    fields = in_tx.split(',')
                    to_address = fields[4]
                    pay_timestamp = fields[1]

                    if to_address not in self.receivers:
                        self.receivers[to_address] = pay_timestamp
                    else:
                        if pay_timestamp < self.receivers[to_address]:
                            self.receivers[to_address] = pay_timestamp

                    if to_address not in self.payment_counts:
                        self.payment_counts[to_address] = 1
                    else:
                        self.payment_counts[to_address] += 1

                counter += 1
            self.payments_num = counter - 1
            payment_count_list = [self.payment_counts[address] for address in self.receivers]
            self.maxpay = max(payment_count_list) if len(payment_count_list) > 0 else 0
            # print('maxpay is {%d}' % self.maxpay)
            # print(self.receivers)
            # print('payments_num is {%d}' % self.payments_num)

        pay_after_investment_counter = 0
        for address in self.receivers:
            if address in self.investors and self.receivers[address] > self.investors[address]:
                pay_after_investment_counter += 1
        self.returnrio = pay_after_investment_counter / len(self.receivers) if len(self.receivers) > 0 else 0
        # print('returnrio is {%f}' % self.returnrio)

        get_paid_investors_counter = 0
        for address in self.investors:
            if address in self.receivers:
                get_paid_investors_counter += 1
        self.Pr = get_paid_investors_counter / len(self.receivers) if len(self.receivers) > 0 else 0
        # print('Pr is {%f}' % self.Pr)

        # get the balance of a tx from the file flaged.csv

    def get_balance(self):
        with open('./data/flaged.csv', 'r') as flaged_csv:
            for contract in flaged_csv:
                if self.contract_address == contract.strip().split(',')[0]:
                    self.bal = float(contract.strip().split(',')[3].split(' ')[0])
        # print('bal is {%f}' % self.bal)

    def get_action_frequency(self):
        with open('./data/contracts/' + self.ponzi_or_nonponzi + '/' + self.contract_address + '.txt',
                  'r') as code_file:
            code = code_file.read()
            self.code_processing(code)

    def code_processing(self, text):
        text = re.sub('[\[!@#$\]]', '', text)
        lines = text.split(',')[:-5]
        for line in lines:
            # print(line.strip())
            line_action = line.strip().split(' ')[1]
            if line_action not in self.action_freq:
                self.action_freq[line_action] = 1
            else:
                self.action_freq[line_action] += 1

        total_num_of_actions = sum(self.action_freq.values())
        self.action_ratio['GASLIMIT'] = self.action_freq['GASLIMIT'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'GASLIMIT' in self.action_freq else 0

        self.action_ratio['EXP'] = self.action_freq['EXP'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'EXP' in self.action_freq else 0

        self.action_ratio['CALLDATALOAD'] = self.action_freq['CALLDATALOAD'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'CALLDATALOAD' in self.action_freq else 0

        self.action_ratio['SLOAD'] = self.action_freq['SLOAD'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'SLOAD' in self.action_freq else 0

        self.action_ratio['CALLER'] = self.action_freq['CALLER'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'CALLER' in self.action_freq else 0

        self.action_ratio['LT'] = self.action_freq['LT'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'LT' in self.action_freq else 0

        self.action_ratio['GAS'] = self.action_freq['GAS'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'GAS' in self.action_freq else 0

        self.action_ratio['MOD'] = self.action_freq['MOD'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'MOD' in self.action_freq else 0

        self.action_ratio['MSTORE'] = self.action_freq['MSTORE'] * 100 / total_num_of_actions if len(
            self.action_freq) > 0 and 'MSTORE' in self.action_freq else 0

        # print(self.action_freq)
        # print(self.action_ratio)

    def get_d_ind(self):
        all_participants = {}
        for investor in self.investment_count:
            n_i = self.payment_counts[investor] if investor in self.payment_counts else 0
            m_i = self.investment_count[investor]
            all_participants[investor] = n_i - m_i
        for payer in self.payment_counts:
            if payer not in self.investment_count:
                n_i = self.payment_counts[payer]
                m_i = 0
                all_participants[payer] = n_i - m_i

        res = all(x == 0 for x in all_participants.values())
        if res or len(all_participants) <= 2:
            self.D_ind = 0
        else:
            v_list = list(all_participants.values())
            mean_of_v_list = mean(v_list)
            median_of_v_list = median(v_list)
            std_of_v_list = stdev(v_list)
            skewness = 3 * (mean_of_v_list - median_of_v_list) / (std_of_v_list) if std_of_v_list != 0 else 0
            self.D_ind = skewness



if __name__ == '__main__':

    # read ponzi_Contracts.csv and non_ponziContracts.csv file and find the balance of each contract by looking into
    contractFeature = ContractFeature('0xece701c76bd00d1c3f96410a0c69ea8dfcf5f34e', 'nonponzi')
