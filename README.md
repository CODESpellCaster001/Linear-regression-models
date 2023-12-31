# Linear-regression-models
Predict net energy output (PE) in a power plant using linear regression with AT, V, AP, RH as features.

main task:
This dataset contains 9568 data points collected from combined cycle power plants over 6 years (2006-2011) when the plants were set to operate at full capacity. Features include hourly average environmental variables temperature (T), ambient pressure (AP), relative humidity (RH), and exhaust vacuum (V) to predict the plant's hourly net electrical energy output (EP).
A combined cycle power plant (CCPP) consists of a gas turbine (GT), a steam turbine (ST) and a heat recovery steam generator. In CCPP, electricity is generated by gas turbines and steam turbines, which are combined in a cycle and transferred from one turbine to the other. While vacuum comes from and affects the steam turbine, three other environmental variables affect gas turbine performance.
To be comparable to our baseline study and to allow for 5x2-fold statistical testing, we present five times shuffled data. For each shuffle, a 2-fold CV was performed and the resulting 10 measurements were used for statistical testing.
We provide data in .ods and .xlsx formats.
Relevant documents cited:
[1] Pınar Tüfekci, Prediction of full-load power output of combined cycle power plants operating at base load using machine learning methods, International Journal of Power and Energy Systems, Volume 60, September 2014, Pages 126-140, ISSN 0142 -0615, http://dx.doi.org/10.1016/j.ijepes.2014.02.027.
(http://www.sciencedirect.com/science/article/pii/S0142061514000908)
[2] Heysem Kaya, Pınar Tüfekci, Sadık Fikret Gürgen: Local and global learning methods for predicting combined gas and steam turbine power, Proceedings of the International Conference on Emerging Trends in Computer and Electronic Engineering, ICETCEE 2012, pp. 13-18 (2012 Dubai, March 2019)

This is a cycle power plant data with a total of 9568 sample data, each data has 5 columns: AT (temperature), V (pressure), AP (humidity), RH (pressure), PE (output power).
Our problem is to get a linear relationship, corresponding to PE is the sample output, and A T , V , AP , and R H are the sample features. The purpose of machine learning is to get a linear regression model, that is:
 PE = θ_0 + θ_1 * AT + θ_2 * V + θ_3 * AP + θ_4 * RH
What needs to be learned are the four parameters θ_0, θ_1, θ_2, θ_3, and θ_4.
