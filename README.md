# workload_prediction
I am making the data public used in the following publication:

R. Mian, et al., "Towards Building Performance Models for Data-intensive Workloads in Public Clouds", in 4th ACM/SPEC Intl. Conf. on Performance Engineering (ICPE), Prague, Czech Republic, 2013, pp. 259-270.

https://pdfs.semanticscholar.org/43bc/4d73772bb5770ef6af32353d2069be01b0fc.pdf

Abstract:
resources and, therefore, becomes a promising candidate for large-scale data-intensive computing. In this paper, we explore experiment-driven performance models for data-intensive workloads executing in an infrastructure-as-a-service (IaaS) public cloud. The performance models help in predicting the workload behaviour, and serve as a key component of a larger framework for resource provisioning in the cloud. We determine a suitable prediction technique after comparing popular regression methods. We also enumerate the variables that impact variance in the workload performance in a public cloud. Finally, we build a performance model for a multi-tenant data service in the Amazon cloud. We find that a linear classifier is sufficient in most cases. On a few occasions, a linear classifier is unsuitable and non-linear modeling is required, which is time consuming. Consequently, we recommend that a linear classifier be used in training the performance model in the first instance. If the resulting model is unsatisfactory, then non-linear modeling can be carried out in the next step.

