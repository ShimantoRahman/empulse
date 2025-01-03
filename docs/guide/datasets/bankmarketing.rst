.. _upsell_bank_telemarketing:

=======================
Bank Marketing Data Set
=======================

This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Abstract: The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).
	
Source:

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

Data Set Information:

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

There are four datasets:
1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

Feature Description
===================

.. list-table::
   :header-rows: 1

   * - Variable Name
     - Description
     - Type
   * - age
     - Age of the client
     - numeric
   * - job
     - Type of job (e.g., 'admin.', 'blue-collar', 'entrepreneur', etc.)
     - categorical
   * - marital
     - Marital status (e.g., 'divorced', 'married', 'single', etc.)
     - categorical
   * - education
     - Education level (e.g., 'basic.4y', 'high.school', 'university.degree', etc.)
     - categorical
   * - default
     - Has credit in default? ('no', 'yes', 'unknown')
     - categorical
   * - housing
     - Has housing loan? ('no', 'yes', 'unknown')
     - categorical
   * - loan
     - Has personal loan? ('no', 'yes', 'unknown')
     - categorical
   * - contact
     - Contact communication type ('cellular', 'telephone')
     - categorical
   * - month
     - Last contact month of year (e.g., 'jan', 'feb', 'mar', etc.)
     - categorical
   * - day_of_week
     - Last contact day of the week (e.g., 'mon', 'tue', 'wed', etc.)
     - categorical
   * - duration
     - Last contact duration, in seconds
     - numeric
   * - campaign
     - Number of contacts performed during this campaign and for this client
     - numeric
   * - pdays
     - Number of days that passed by after the client was last contacted from a previous campaign
     - numeric
   * - previous
     - Number of contacts performed before this campaign and for this client
     - numeric
   * - poutcome
     - Outcome of the previous marketing campaign ('failure', 'nonexistent', 'success')
     - categorical
   * - emp.var.rate
     - Employment variation rate - quarterly indicator
     - numeric
   * - cons.price.idx
     - Consumer price index - monthly indicator
     - numeric
   * - cons.conf.idx
     - Consumer confidence index - monthly indicator
     - numeric
   * - euribor3m
     - Euribor 3 month rate - daily indicator
     - numeric
   * - nr.employed
     - Number of employees - quarterly indicator
     - numeric
   * - y
     - Has the client subscribed a term deposit? ('yes', 'no')
     - binary

Relevant Papers:

S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimaraes, Portugal, October, 2011. EUROSIS. [bank.zip]


Citation Request:

This dataset is public available for research. The details are described in [Moro et al., 2014].
Please include this citation if you plan to use this database:

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
