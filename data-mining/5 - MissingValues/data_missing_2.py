import random

def replace_with_percentage(line, percentage):
    modified_line = []
    for item in line.split(','):
        if random.random() < percentage:
            modified_line.append('?')
        else:
            modified_line.append(item)
    return ','.join(modified_line)

def modify_database(input_filename, output_filename, percentage_to_replace):
    with open(input_filename, 'r') as input_file:
        data = input_file.readlines()

    modified_data = [replace_with_percentage(line.strip(), percentage_to_replace) for line in data]

    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join(modified_data))

# Defina o nome do arquivo de entrada e saída, e a porcentagem de substituição desejada
input_filename = 'krkopt.data'
output_filename = 'krkopt_missing_values.data'
percentage_to_replace = 0.08  # 20%

modify_database(input_filename, output_filename, percentage_to_replace)

print(f"Os dados foram modificados e salvos em '{output_filename}'.")
