import os
import sys
import csv


class OpcodeColector(object):
    def __init__(self, ponzi_contract_filename, non_ponzi_filename, opcode_filename):
        self.ponzi_filename = ponzi_contract_filename
        self.non_ponzi_filename = non_ponzi_filename
        self.opcode_filename = opcode_filename

        self.get_opcode_and_save()

        self.get_opcode_and_save_non_ponzi()

    def search_opcode_by_contract_address(self, contract_address):
        opcode = None
        with open(self.opcode_filename, 'r') as opcode_file:
            for row in opcode_file:
                if row.split(',', 1)[0] == contract_address.lower() or row.split(',', 1)[0] == contract_address:
                    opcode = row.split(',', 1)[1]
                    return opcode

    def get_opcode_and_save(self):
        print('=== Downloading ponzi contract files...')
        with open(self.ponzi_filename, 'r') as ponzi_file:
            csv_reader = csv.reader(ponzi_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if 0 < line_count:
                    print(row[1])
                    opcode = self.search_opcode_by_contract_address(row[1])
                    if opcode is not None:
                        file_name_to_save = './data/contracts/ponzi/' + \
                            row[1] + '.txt'
                        if not os.path.exists(file_name_to_save):
                            with open(file_name_to_save, 'w+') as f:
                                f.write(" ")
                        with open(file_name_to_save, 'w+') as write_file:
                            write_file.write(opcode)
                line_count += 1
                if line_count % 200 == 0 and line_count > 0:
                    print(
                        '{0} ponzi contracts have downloaded...'.format(line_count))
        print('ponzi contracts downloading is over.')

    def get_opcode_and_save_non_ponzi(self):
        print('=== Downloading nonponzi contract files...')
        with open(self.non_ponzi_filename, 'r') as non_ponzi_file:
            csv_reader = csv.reader(non_ponzi_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if 0 < line_count:
                    print(row[1])
                    opcode = self.search_opcode_by_contract_address(row[1])
                    if opcode is not None:
                        file_name_to_save = './data/contracts/nonponzi/' + \
                            row[1] + '.txt'
                        if not os.path.exists(file_name_to_save):
                            with open(file_name_to_save, 'w+') as f:
                                f.write(" ")
                        with open(file_name_to_save, 'w+') as write_file:
                            write_file.write(opcode)
                line_count += 1
                if line_count % 200 == 0 and line_count > 0:
                    print(
                        '{0} nonponzi contracts have downloaded...'.format(line_count))
        print('nonponzi contracts downloading is over.')


if __name__ == '__main__':
    opcode_collector = OpcodeColector(
        './ponziContracts.csv', './non_ponziContracts.csv', './Opcodes.csv')
