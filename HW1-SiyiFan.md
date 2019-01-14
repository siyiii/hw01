
# Homework 01: Model Building and Model Selection/Fitting
## Siyi Fan


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numpy import meshgrid

%matplotlib inline
```

## Building models - Deviant aggressive behavior
1. For the first theory, social policy that could punish individuals who show deviant aggressive behaviors would be appropriate (e.g, go to the jail) and reward is the avoidance of the punishment or report of aggressive behaviors. Besides, the policy should clearly clarify the boundary of deviant aggressive behaviors, that is, what kinds of behaviors break the law. For the second theory, students could acquire emotion management skills under policy related to education or school; also, social policy that could provide psychological or clinical treatment would work. When an individual is frustrated in his personal life, he could turn to specific organization that helps him successfully control feelings and stay positive. For the third theory, Non-Discrimination policy could legally protect oppressed individuals and ensure each individual is treated with respect and equality. For the fourth theory, the policy that ensures access to community-based treatment facilities, safer environments, sensitivity training for the police and other criminal justice officials would be appropriate. Early intervention would also work, because it allows for community leaders to devote additional attention to those at-risk youth early on so that prevent them becoming the subculture roles. 

2. Take #MeToo movement for an example. Here, deviant aggressive behaviors include sexual harrassment, sexual abuse, and other sexual violence. For the first theory, the movement punishs offenders using social media so that their reputations can be destroyed in an instant and the law will then be brought into to investigate the crime. The reward would be the absence of the punishment. For the second theory, most offenders are males who want to show their masculine power. If one male loses control over his own life, he may capture all things that could make themselves powerful and sexual misconduct would be a way to express their anger and regain confidence. Males should also get help when needed and society should accept males' failure. By taking psychological treatment and education on gender equlity, males could manage their emotions in an appropriate way and hopefully encourage other men to come forward so they can also heal. For the third theory, females are oppressed group in most sexual misconduct cases. Non-Discrimination policy would be important to ensure females get respect and equal treatment (such as equality in pay) as males get. Also, the awareness of gender equality should be raised on at a young age for both males and females. Males need to show respect for women and inspire other males to stand for gender equality; females need to realize how to fight for our rights and help each other overcome inequality. For the fourth theory, a culture that encourages a trend of toxic masculinity (support gender inequality and abuse of power) could be viewed as a subculture in our society. #MeToo movement to some extent pushes this culture shift and dissolves the power of culture that is proud of celebrating sexual misconduct. On one hand, members in this subculture may worry their reputations get destroyed, so they withdraw from the group; on the other hand, people who have awareness of gender equality or have courage to speak out would suppress those privileged males hold the most power. 

## Building models - Waiting until the last minute
1. First, people may find the tasks boring and negative emotion (e.g. frustration) could diminish individual's motivation and concentration to finish the tasks on time; second, people who pursue perfectionism may delay their work and even keep themselves from getting started; third, people may underestimate how long it will take them to complete the task and overconfidence will lead them to start the task at the last minute; finally, people tend to procrastinate the task when its deadline is far away.
2. If there is a list of intended tasks and we let P (Priority) be the place of one task on this list, then P is equal to the Importance of this task (the urgency of this task) * {(Negative emotions towards the tasks + the costs of taking the task) / The estimated time to complete the task}. "Importance of this task (the urgency of this task)" is an external variable and it is not relevant to the personal factors or internal variable leading to procrastination, so it could be treated as a weight to the internal variables. If the sum of "Negative emotions to the tasks" and "the costs of taking the task" is high and the value of "The estimated time to complete the task" is small (overconfidence), then the P (Priority) will be large and more likely to procrastinate. 
3. P (Priority) may also be equal to the Perfectionism * (the costs of taking the task/The estimated time to complete the task). "Perfectionism" could be viewed as a weight. The higher tendency of perfectionism and the shorter estimated time would lead to larger P and increase the likelihood of procrastination. 

Note: "the costs of taking the task" could be viewed as a pleasure level. For example, doing homework would be a huge cost of pleasure, so the value of "the costs of taking the task" would be large; while watching movie would be a small cost of pleasure, so the value of "the costs of taking the task" would be small. In this way, if a task is more enjoyable than other tasks, people may put it as a top priority so that the value of P (Priority) would be small. 
4. For the first model, H1: people who suffer from mental illnesses (e.g., depression or anxiety) tend to delay taking action. The reason is that those people always have negative emotions which diminishes their motivation and perseverance on tasks. In this way, the value of "Negative emotions towards the tasks" will be large and increase the outcome of P (Priority). H2: Overconfident individuals tend to delay taking action. This kind of people may overestimate their ability to complete the task, so they may start later than they think they will need, which leads to a large P (Priority). For the second model, H1: People who think “It has to be perfect" may delay taking action, because this overly demanding standard would boost the value of "Perfectionism" so that increase P (Priority). H2: People with high self-control will take action early on. This kind of people could focus on the gains of future and deemphasize the frustration of the present, so the value of "the costs of taking the task" would be small, which leads to a smaller P (Priority). 

## Selecting and fitting a model - 1

a. Large sample size allows the model to fit more parameters, find the nonlinear effect and finally improves the generalizability and estimation. 
b. Flexible learning method would perform worse because it captures all the noise for the training data instead of finding the signal. In this way, this overfit model will then make predictions based on that noise. Therefore, it will perform well on its training data yet poorly on the new data. 
c. Flexible learning method would perform better because it uses various groups of data to generate wider range of possible shapes, increasing the likelihood of finding non-linear shape.
d. Flexible learning method would perform worse because it will include noise and not be able to generalize to a new group of data. 

## Selecting and fitting a model - 2a


```python
x = np.linspace(0, 10, 10)

bias = 100 * np.cos(x/3) + 100
irreducible_error = 0 * x + 100
train_error = 150 * np.cos(x/3) + 60
variance = x**2/3
test_error = bias + variance + irreducible_error

plt.plot(bias, color="green", label="Bias")
plt.plot(irreducible_error, color="grey", label="Irreducible error")
plt.plot(test_error, color="orange", label="Test error")
plt.plot(train_error, color="yellow", label="Train error")
plt.plot(variance, color="darkblue", label="Variance")
plt.legend(["Bias", "Irreducible error", "Test error", "Train error", "Variance"], bbox_to_anchor=(1.5,1))
plt.xlabel("Flexibility")
plt.ylabel("Error rate/MSE")
plt.show()
```


![png](output_7_0.png)


## Selecting and fitting a model - 2b
1. Bias characterises the difference between the averages of the estimate and the true values. Bias will decrease with higher flexibility because higher flexibility could fit the training data easily so that capture the underlying behavior of the true functional form well. 
2. Variance determines how much the average model estimation deviates as different training data is tried. Variance will increase with higher flexibility because higher flexibility or different data points has impact on the model (parameters) fitting. 
3. Training error will always decrease because flexible model could explain the variation in the training dataset so that minimize the difference between the estimate and the true training values. The model will completely fit the training data when the training error goes to zero. 
4. Test error is always U-shaped because it is the sum of bias, variance and irreducible error. Generally, as flexibility increases we see an increase in variance and a decrease in bias. As flexibility is increased, the bias tends to drop quickly (faster than the variance increases) and so we see a drop in test error. However, as flexibility increases further, the test error goes up because there is less reduction in bias and instead the variance increases quickly due to overfitting.
5. Irreducible error will always exist no matter how well we estimate the parameters.

## Selecting and fitting a model - 3


```python
# simulate random uniform values of x1 and x2
x1 = np.random.uniform(-1, 1, 200)
x2 = np.random.uniform(-1, 1, 200)

# function to calculate Bayes decision rule
def bayes_rule(x1, x2):
    return x1 + x1**2 + x2 + x2**2

# classify the test data
logodds = bayes_rule(x1, x2) + np.random.normal(0, 0.5, 200)
y = logodds
def prob(y):
    return np.exp(y)/(1 + np.exp(y))
success = prob(y) > 0.5

# plot Bayes decision boundary
X, Y = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
def bound(X, Y):
    return prob(bayes_rule(X,Y)) > 0.5

plt.scatter(x1[success], x2[success], color="grey", label="Success")
plt.scatter(x1[np.invert(success)], x2[np.invert(success)], color="green", label="Failure")
plt.contour(X, Y, bound(X, Y), 1)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend(["Success", "Failure"], bbox_to_anchor=(1.5,1))
plt.title("Bayes Classifer")
plt.grid(True)
plt.show()
```


![png](output_10_0.png)



```python

```
