# Predicting Equity Volatilty using Sentiment and Technical Analysis

### To set up a basic environment
1.  Run `pip install -r requirements.txt` to download the necessary Python modules
2.  Run `python data.py` to download & prepare the data
3. Data used for this project can be found in `symbol_fundamentals.csv`, `tweets*/`, `equities/`, `database.db` (SQLite), and `applied_stocks.py` (list of stock symbols we analyzed)
3. Run `python supervised_ensemble.py` to perform the supervised learning analysis
4. Run `final_model.py` to view the final version of the supervised model
5. The supervised model results and their visualizations can be viewed directly in `supervised.ipynb`, `hyper.csv`, `mae.csv`, `mse.csv`, `aggregate.csv`, `ensemble.pkl`, `results.pkl` and `pickles/`
6. The unsupervised exploration and analysis can be found in `analysis_of_clusters.ipynb`, `cluster_analysis.ipynb`, and `cluster_techs.ipynb`
7. The resulting clusters from our unsupervised analysis can be viewed directly in `stock_clusters.csv`

### For any questions, please contact a member of the project team.