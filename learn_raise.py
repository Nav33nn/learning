


def method_with_raise():
	try:
		kajsdh
		print('try')
	except:
		print('raising error')
		raise ValueError('string error')



def run():
	try:
		a = method_with_raise()
	except ValueError as e:
		print('in except')
		print(str(e))


if __name__ == '__main__':
	run()