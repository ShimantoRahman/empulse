.PHONY: clean build

clean:
	powershell Remove-Item -Recurse -Force dist\*
	powershell Remove-Item -Recurse -Force build\*
	powershell Remove-Item -Recurse -Force *.egg-info\

build: clean
	python setup.py sdist bdist_wheel

upload: build
	twine upload dist/*