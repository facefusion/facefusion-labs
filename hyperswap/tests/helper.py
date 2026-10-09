from configparser import ConfigParser


def create_test_config_parser(output_size : int) -> ConfigParser:
	config_parser = ConfigParser()
	config_parser.read_dict(
	{
		'training.model.generator':
		{
			'source_channels': '512',
			'output_size': str(output_size),
			'num_blocks': '2'
		},
		'training.model.masker':
		{
			'input_channels': '67',
			'output_channels': '1',
			'num_filters': '16'
		}
	})
	return config_parser
