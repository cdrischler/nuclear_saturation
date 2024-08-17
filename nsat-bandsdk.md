# BAND SDK v0.2 Community Policy Compatibility for `nuclear_saturation`


> This document summarizes the efforts of current and future BAND member packages to achieve compatibility with the BAND SDK community policies.  Additional details on the BAND SDK are available [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/bandsdk.md) and should be considered when filling out this form. The most recent copy of this template exists [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/template.md).
>
> This file should filled out and placed in the directory in the `bandframework` repository representing the software name appended by `bandsdk`.  For example, if you have a software `foo`, the compatibility file should be named `foobandsdk.md` and placed in the directory housing the software in the `bandframework` repository. No open source code can be included without this file.
>
> All code included in this repository will be open source.  If a piece of code does not contain a open-source LICENSE file as mentioned in the requirements below, then it will be automatically licensed as described in the LICENSE file in the root directory of the bandframework repository.
>
> Please provide information on your compatibility status for each mandatory policy and, if possible, also for recommended policies. If you are not compatible, state what is lacking and what are your plans on how to achieve compliance. For current BAND SDK packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future BAND member packages.
>
> To suggest changes to these requirements or obtain more information, please contact [BAND](https://bandframework.github.io/team).

<!-- #region -->
**Website:** https://github.com/cdrischler/nuclear_saturation/   
**Contact:** Christian Drischler (<drischler@ohio.edu>)    
**Icon:** https://github.com/cdrischler/nuclear_saturation/blob/main/logo/logo_alternative.png    
**Description:** `Nuclear_satuation` provides a Bayesian mixture model approach with conjugate distributions to extract the *empirical* nuclear saturation point from a range of Density Functional Theory (DFT) constraints, including those from relativistic mean field (RMF) theory and Skyrme energy density functionals. 
Its aims to provide statistically meaningful constraints on the nuclear saturation point to benchmark (and guide the construction of novel) microscopic interaction derived from chiral effective field theory (EFT).

### Mandatory Policies

**BAND SDK**

| # | Policy                 |Support| Notes                                                                                                                                                        |
|---|-----------------------|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. | Support BAND community GNU Autoconf, CMake, or other build options. |Full| The source code is written in Python, so it does not need `Cmake` or `Autoconf`                                                                              
| 2. | Have a README file in the top directory that states a specific set of testing procedures for a user to verify the software was installed and run correctly. |Full| None.                                                                                                                                                        |
| 3. | Provide a documented, reliable way to contact the development team. |Full| The contact is <drischler@ohio.edu>, given in the top-level `README.md`.                                                                                     |
| 4. | Come with an open-source license |Full| Uses the MIT license.                                                                                                                                        |
| 5. | Provide a runtime API to return the current version number of the software. |Full| None.                                                                                                                                                        |
| 6. | Provide a BAND team-accessible repository. |Full| The current public repository at https://github.com/cdrischler/nuclear_saturation/ will be publicly available and accessible by anyone in BAND or otherwise. |
| 7. | Must allow installing, building, and linking against an outside copy of all imported software that is externally developed and maintained .|Full| None.                                                                                                                                                        |
| 8. |  Have no hardwired print or IO statements that cannot be turned off. |Full| None.                                                                                                                                                        |


### Recommended Policies

| # | Policy                 | Support | Notes                                                                                                          |
|---|------------------------|---------|----------------------------------------------------------------------------------------------------------------|
|**R1.**| Have a public repository. | Full    | None.                                                                                                          |
|**R2.**| Free all system resources acquired as soon as they are no longer needed. | Full    | None.                                                                                                          |
|**R3.**| Provide a mechanism to export ordered list of library dependencies. | None    | None.                                                                                                          |
|**R4.**| Document versions of packages that it works with or depends upon, preferably in machine-readable form.  | Partial | The repository contains install instructions for install virtual environments and installing the dependencies. |
|**R5.**| Have SUPPORT, LICENSE, and CHANGELOG files in top directory.  | Partial | Possesses LICENSE file in the top directory.                                                                   |
|**R6.**| Have sufficient documentation to support use and further development.  | Partial | Possesses docstrings in the source code.                                                                       |
|**R7.**| Be buildable using 64-bit pointers; 32-bit is optional. | Full    | Package supports both 32 and 64 bit under same API.                                                            |
|**R8.**| Do not assume a full MPI communicator; allow for user-provided MPI communicator. | N/A     | None.                                                                                                          |
|**R9.**| Use a limited and well-defined name space (e.g., symbol, macro, library, include). | Full    | None.                                                                                                          |
|**R10.**| Give best effort at portability to key architectures. | Full    | None.                                                                                                          |
|**R11.**| Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively. | N/A     | None.                                                                                                          |
|**R12.**| All BAND compatibility changes should be sustainable. | Full    | None.                                                                                                          |
|**R13.**| Respect system resources and settings made by other previously called packages. | Full    | None.                                                                                                          |
|**R14.**| Provide a comprehensive test suite for correctness of installation verification. | Partial | The repository possesses a basic `pytest` suite to test for simple errors.                                     |
<!-- #endregion -->
