from baselines.han import add_han_specific_parser, HAN

'''
This class acts as factory of all deep models.
'''

parser_maps = {
    'han': add_han_specific_parser
}

model_maps = {
    'han': HAN
}
