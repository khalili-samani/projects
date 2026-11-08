SHELL := /bin/bash
MYSQL ?= mysql
MYSQL_ARGS ?= -h localhost -P 3306 -u housing_user -phousing_password
.PHONY: setup load transform test analyse all
setup:
	$(MYSQL) $(MYSQL_ARGS) < sql/run_all.sql
load:
	python scripts/load_data.py
transform:
	$(MYSQL) $(MYSQL_ARGS) < sql/run_transform.sql
test:
	$(MYSQL) $(MYSQL_ARGS) < sql/08_tests/01_data_quality_tests.sql
analyse:
	$(MYSQL) $(MYSQL_ARGS) < sql/07_analysis/01_business_questions.sql
all: setup load transform test analyse
