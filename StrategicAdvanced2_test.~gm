$GDXIN in.gdx

set t,s,tech,d;

parameters
K_co2(d,tech,t,s)                         coefficients for CO2 savings
K_opex(d,tech,t,s)                        coefficients for Opex savings
K0_capex(tech,t)                    coefficients for capex savings
K1_capex(tech,t)                    coefficients for capex savings
CO2_savingTarget(t)
IO_modular(tech)
x_limit_bot_opex(d,tech,t,s)
x_limit_top_opex(d,tech,t,s)
x_limit_bot_co2(d,tech,t,s)
x_limit_top_co2(d,tech,t,s)
;

*to be done automatically trough python
$OnEps
$load t
$load s
$load tech
$load d
$load K_co2, K_opex, K0_capex, K1_capex, CO2_savingTarget, IO_modular, x_limit_bot_opex, x_limit_top_opex, x_limit_bot_co2, x_limit_top_co2




positive variables
x(tech,t,s)                     size of each technology
opexSavings(t,s)
co2Savings(t,s)
capex(t,s)
Inv_year
non_modular_capex(tech,t,s)
;

variables
z
;

Binary variables
IO_installation(tech,t,s)                  if a unit is installed or not
IO_cons_co2(d,t,s)
IO_cons_opex(d,t,s)
;

Equations
Obj
Cal_opexSavings1(d,t,s)
Cal_opexSavings2(d,t,s)
Cal_opexSavings3(d,tech,t,s)
Cal_opexSavings4(d,tech,t,s)
Cal_opexSavings5(t,s)

Cal_co2Savings1(d,t,s)
Cal_co2Savings2(d,t,s)
Cal_co2Savings3(d,tech,t,s)
Cal_co2Savings4(d,tech,t,s)
Cal_co2Savings5(t,s)

Cal_capex(t,s)

Con_co2Savings(t)
Con_size_increase(tech,t,s)
Con_size_0(tech,t,s)
Con_Inv_year(t)
Con_IO_installation(tech,t,s)
Con_non_modular_capex(tech,t,s)
;



Obj..                     z =E=  sum(t, Inv_year)-sum(t,sum(s,opexSavings(t,s)));

Cal_opexSavings1(d,t,s)..        opexSavings(t,s) =G= sum(tech,  K_opex(d, tech,t,s)*x(tech,t,s)) - (1-IO_cons_opex(d,t,s))*1000000;
Cal_opexSavings2(d,t,s)..        opexSavings(t,s) =L= sum(tech,  K_opex(d, tech,t,s)*x(tech,t,s)) + (1-IO_cons_opex(d,t,s))*1000000;
Cal_opexSavings3(d,tech,t,s)..   x(tech,t,s) =G= x_limit_bot_opex(d,tech,t,s) - (1-IO_cons_opex(d,t,s))*1000000;
Cal_opexSavings4(d,tech,t,s)..   x(tech,t,s) =L= x_limit_top_opex(d,tech,t,s) + (1-IO_cons_opex(d,t,s))*1000000;
Cal_opexSavings5(t,s)..          sum(d, IO_cons_opex(d,t,s)) =e= 1;

Cal_co2Savings1(d,t,s)..        co2Savings(t,s) =G= sum(tech,  K_co2(d, tech,t,s)*x(tech,t,s)) - (1-IO_cons_co2(d,t,s))*1000000;
Cal_co2Savings2(d,t,s)..        co2Savings(t,s) =L= sum(tech,  K_co2(d, tech,t,s)*x(tech,t,s)) + (1-IO_cons_co2(d,t,s))*1000000;
Cal_co2Savings3(d,tech,t,s)..   x(tech,t,s) =G= x_limit_bot_co2(d,tech,t,s) - (1-IO_cons_co2(d,t,s))*1000000;
Cal_co2Savings4(d,tech,t,s)..   x(tech,t,s) =L= x_limit_top_co2(d,tech,t,s) + (1-IO_cons_co2(d,t,s))*1000000;
Cal_co2Savings5(t,s)..          sum(d, IO_cons_co2(d,t,s)) =e= 1;

Cal_capex(t,s)..          capex(t,s)  =E= sum(tech,  K1_capex(tech,t)*(x(tech,t,s)   - x(tech,t-1,s))  +  non_modular_capex(tech,t,s) );


Con_Inv_year(t)..                          sum(s, capex(t,s)) =l=  Inv_year;
Con_co2Savings(t)..                        sum(s,co2Savings(t,s))  =g=   CO2_savingTarget(t);
Con_size_increase(tech,t,s)..              x(tech,t,s)=g=x(tech,t-1,s);
Con_size_0(tech,t,s)..                     x('dummy',t,s)=e=1;
Con_IO_installation(tech,t,s)..            IO_installation(tech,t,s) =g= (x(tech,t,s)-x(tech,t-1,s))*0.0001 - IO_modular(tech)*1000000;
Con_non_modular_capex(tech,t,s)..          non_modular_capex(tech,t,s) =g=   (K0_capex(tech,t) + K1_capex(tech,t)*x(tech,t,s)) - (1-IO_installation(tech,t,s))*1000000;


Model Stratey  /all/;

*psi.l(t) = part_load(t);
Option Optcr=0.001;
Solve Stratey using mip minimizing z;

*psi.l(t)$(psi.l(t)=0) = EPS;

execute_unload "output.gdx"   z  x co2Savings  capex   Inv_year   IO_installation IO_cons_opex
*execute "gdx2sqlite -i output.gdx -o output.db";
