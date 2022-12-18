/*INCLUDE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ran2.h"

//COSTANTS
#define N              100
#define LINK           3
#define phi	       0.1
#define alpha           0.3
#define delta          0.015
#define T              1000
#define depositRate    0.03
#define riskfree       0.01
#define L_entry        600
#define interest_entry 0.02
#define RANDOM_CONNECTIVITY 0.75
#define TOT_SIM 4
#define Bank_exit 0
#define active 0.0 
#define tau 1
#define tauI 1
#define eta 0.09
#define leverage_target 33.333
#define trans_cost 0.0
#define prud 0.045
#define nu 0.01
#define ypsilon 1.00
#define sigma 1.5
#define kost 1.  */

//INCLUDE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ran2.h"

//COSTANTS
#define N              100     
#define S              10
#define LINK           3
#define phi	       0.1
#define alpha          /*0.07*/ 0.3
#define delta         0.05        // 0.045     //  0.015
#define T              1000
#define depositRate    0.03
#define riskfree       0.01
#define L_entry        600
#define interest_entry 0.02
#define RANDOM_CONNECTIVITY 0.75
#define TOT_SIM 10
#define Bank_exit 0
#define active 0.0 
#define tau 1
#define tauI 1
#define eta 0.09
#define leverage_target 33.333
#define trans_cost 0.0
#define prud     0.02           //0.015     //0.045
#define nu 0.01
#define ypsilon 1.0
#define sigma 1.5
#define kost 1.
//#define Price 0.3
//GLOBALS



int i, j, t,z, dayc, shokkati, n_sim, indegree_max, guru,guru2, guru3, guru4, guru5,guru6,guru7,guru8, guru9,guru10,iT_inv, incominglinkMax,i_max, i_max2,num,num1, coretot, peripherytot, inlinkcore, inlinkperi, totmorticore, totmortiperi, thre;

float T_inv, leverageMax, equityMax,incominglinkperc, liquidityMax, threshold, meanlinkcore, meanlinkperi, meanintcore, meanintperi, meaninterestcore[N], meaninterestperi[N], meanmorticore, meanmortiperi, meanbaddebtcore, meanbaddebtperi, totlevacore, totlevaperi, meanlevacore, meanlevaperi, totcapacitycore, totcapacityperi, meancapacitycore, meancapacityperi, totrationcore, totrationperi, meanrationcore, meanrationperi, totbaddebtcore, totbaddebtperi, thre1 ;


int failB[N], outdegree[N], in_tot[N], vicino[N], connected[N], marketBank[N][N],  totfailures,totrationed, si_connect_fail, no_connect_fail, matched[N][N], executed[N],e[N], outgoinglink[N],incominglink[N], rationed[N],insideinter[N],rateorder[N],collegati[N],fallimento_linked,fallimento_nolinked, chi, nochi,mortiguru2,mortiguru3,mortiguru4,mortiguru5,mortiguru6,mortiguru7,mortiguru8,mortiguru9,mortiguru10, core[N],periphery[N];

double loan[N], liquidity[N], deposit[N], equity[N], new_deposit[N],richiesta[N], deltaD[N], Dloan[N], Dloanintera[N],adequacy[N], firesale[N], crunch[N], interbankloan[N], interbankdebt[N], asset[N], leverage[N], loanintero[N], probfail[N], badDebt[N], recupero[N], fit[N], haircut[N], transaction, loanToRate, MeanRate[T], maxleverage,crunchTot, asked, granted ,Price[T], capacity[N], a[N], b[N], c[N], d[N],sommatasso[N], tasso[N], totloan, totliquidity, totdeposit, totequity, totbaddebt, totnewdeposit, totrichesto,totconcesso,totasset, probfaillinked,probfailnonlinked, intradayleverage[N];

double interest_interbank[N][N], credit[N][N],concesso[N], residualcredit[N][N], residualcreditintero[N][N], baddebt[N],depositimedi, equitymedia, baddebtmedi, liquiditymedia, assetmedio;

long idum;

int shocked[N]; // in caso si decida di non shockare tutte le banche

FILE *out, *out1,*out2, *out3, *out4, *out5, *out6, *out7, *out8, *out9, *out10, *out11, *out12, *out13, *out14, *out15, *out16, *out17, *out18, *out19, *out20, *out21, *out22,*out23,*out24,*out25,*out26, *out27, *out28 , *out29, *out30, *out31, *out32, *out33, *out34,*out35,*out36,*out37,*out38,*out39, *out40, *out41, *out42, *out43, *out44, *out45, *out46,*out47, *out48, *out49,*out50, *out51, *out52, *out53, *out54,*out55,*out56, *out57, *out58, *out59, *out60, *out62, *out63, *out64, *out65, *out66, *out67, *out68,  *out69, *out70, *out71, *out72, *out73, *out74;

char file_name[100], file_name1[100],file_name2[100], file_name3[100], file_name4[100],file_name5[100],file_name6[100], file_name7[100], file_name8[100],file_name9[100],file_name10[100],file_name11[100],file_name12[100], file_name13[100], file_name14[100], file_name15[100], file_name16[100], file_name17[100], file_name18[100], file_name19[100], file_name20[100], file_name21[100], file_name22[100],file_name23[100],file_name24[100],file_name25[100],file_name26[100], file_name27[100], file_name28[100], file_name29[100], file_name30[100], file_name31[100],file_name32[100],file_name33[100],file_name34[100],file_name35[100],file_name36[100],file_name37[100],file_name38[100],file_name39[100],file_name40[100], file_name41[100], file_name42[100], file_name43[100],file_name44[100],file_name45[100],file_name46[100],file_name47[100],file_name48[100],file_name49[100],file_name50[100], file_name51[100],file_name52[100],file_name53[100],file_name54[100],file_name55[100],file_name56[100], file_name57[100], file_name58[100], file_name59[100], file_name60[100], file_name62[100], file_name63[100], file_name64[100], file_name65[100], file_name66[100], file_name67[100], file_name68[100], file_name69[100], file_name70[100], file_name71[100], file_name72[100], file_name73[100], file_name74[100]; 





/************FUNCTION-HEADERS**********/

float gasdev(long *idum); 

void Initialization(void);

void Matrix_Random(void);

void Matrix_Preferential(void);

void dfs( int n );

void decreasing(void); //dipende dalla scelta del tipo di rete

void Interest_interbank(void);

void ProbBanks(void);

void Trade(void);
 
void Firesale(void);

void NetworkStatistics(void);

void BanksPayBanks(void);

void AggregateStatistics(void);

void NewBanks(void);

/**************MAIN********************/
/**************MAIN********************/

int main (){
  idum     = -55559222;//734
  dayc     = 1;
  iT_inv   = 500;
  T_inv    = (float)iT_inv/10.;
  /*shokkati = ??;*/

  sprintf(file_name,"df_depositmedio_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon );
  sprintf(file_name1,"df_liquidityimax2_T_inv%.2f_ypsilon%.2f" , T_inv,ypsilon);
  sprintf(file_name2,"df_tassoimax2_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name3,"df_imax2perc_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name4,"df_asked_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name5,"df_granted_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name6,"df_meanrate_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name7,"df_totalefallimenti_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name8,"df_agentequity_T_inv%.2f_ypsilon%.2f",T_inv,ypsilon);
 // sprintf(file_name9,"df_nonisolatedfail_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name10,"df_controlloendday_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name11,"df_trade_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name12,"df_agenteinlink_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name13,"df_imaxperc_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name14,"df_totloan_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name15,"df_totliquidity_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name16,"df_totdeposit_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name17,"df_totequity_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name18,"df_totbaddebt_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name19,"df_equitymedia_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name20,"df_fitness_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name21,"df_agentliquidity_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name22,"df_liquidityguru_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name23,"df_tassoguru_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name24,"df_tassoimax_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name25,"df_liquidityimax_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name26,"df_baddebtguru_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name27,"df_baddebdtmedio_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name28,"df_guru_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name29,"df_liquiditymedia_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name30,"df_inlinkguru_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name31,"df_fitguru_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name32,"df_totnewdep_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name33,"df_equityguru_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name34,"df_equityimax_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name35,"df_equityimax2_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name36,"df_baddebtimax_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name37,"df_baddebtimax2_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name38,"df_depositguru_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name39,"df_credito_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name40,"df_idguru_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name41,"df_deathguru_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
 // sprintf(file_name42,"df_groupguru_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name43,"df_totrationed_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name44,"df_razionamento_T_inv%.2f_ypsilon%.2f", T_inv,ypsilon);
  sprintf(file_name45,"df_totasset_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name46,"df_assetmedio_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name47,"df_probabilitylinked_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name48,"df_probabilitynonlinked_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name49,"df_capacity_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name50,"df_liquidityhubs_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name51,"df_leveragehubs_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name52,"df_tassohubs_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name53,"df_subhubs_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name54,"df_mortesubhubs_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name55,"df_razionamentosubhubs_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name56,"df_failure_linked_subhubs_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name57,"df_exantematrix_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name58,"df_effective_matrix_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name59,"df_intradayleverage_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name60,"df_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
 // sprintf(file_name61,"df_namescoreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name62,"df_inlink_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name63,"df_meaninlink_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name64,"df_meanrate_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name65,"df_totmorti_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name66,"df_meanmorti_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name67,"df_totbaddebt_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name68,"df_meanbaddebt_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name69,"df_totleva_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name70,"df_meanleva_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name71,"df_totcapacity_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name72,"df_meancapacity_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name73,"df_totration_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);
  sprintf(file_name74,"df_meanration_coreperiphery_T_inv%.2f_ypsilon%.2f",  T_inv,ypsilon);


  out  = fopen(file_name, "w");
  out1 = fopen(file_name1, "w");
  out2 = fopen(file_name2, "w");
  out3 = fopen(file_name3, "w");
  out4 = fopen(file_name4, "w");
  out5 = fopen(file_name5, "w");
  out6 = fopen(file_name6, "w");
  out7 = fopen(file_name7, "w");
  out8 = fopen(file_name8, "w");
 // out9 = fopen(file_name9, "w");
 // out10 = fopen(file_name10, "w");
  out11 = fopen(file_name11, "w");
  out12 = fopen(file_name12, "w");
  out13 = fopen(file_name13, "w");
  out14 = fopen(file_name14, "w");
  out15 = fopen(file_name15, "w");
  out16 = fopen(file_name16, "w");
  out17 = fopen(file_name17, "w");
  out18 = fopen(file_name18, "w");
  out19 = fopen(file_name19, "w");
  out20 = fopen(file_name20, "w");
  out21 = fopen(file_name21, "w");
 // out22 = fopen(file_name22, "w");
 // out23 = fopen(file_name23, "w");
  out24 = fopen(file_name24, "w");
  out25 = fopen(file_name25, "w");
  //out26 = fopen(file_name26, "w");
  out27 = fopen(file_name27, "w");
 // out28 = fopen(file_name28, "w");
  out29 = fopen(file_name29, "w");
  //out30 = fopen(file_name30, "w");
  //out31 = fopen(file_name31, "w");
  //out32 = fopen(file_name32, "w");
  //out33 = fopen(file_name33, "w");
  //out34 = fopen(file_name34, "w");
  //out35 = fopen(file_name35, "w");
  //out36 = fopen(file_name36, "w");
  //out37 = fopen(file_name37, "w");
  //out38 = fopen(file_name38, "w");
  out39 = fopen(file_name39, "w");
  //out40 = fopen(file_name40, "w");
  //out41 = fopen(file_name41, "w");
  //out42 = fopen(file_name42, "w");
  out43 = fopen(file_name43, "w");
  out44 = fopen(file_name44, "w");
  out45 = fopen(file_name45, "w");
  out46 = fopen(file_name46, "w");
  //out47 = fopen(file_name47, "w");
  //out48 = fopen(file_name48, "w");
  out49 = fopen(file_name49, "w");
  //out50 = fopen(file_name50, "w");
  //out51 = fopen(file_name51, "w");
  //out52 = fopen(file_name52, "w");
  //out53 = fopen(file_name53, "w");
  //out54 = fopen(file_name54, "w");
  //out55 = fopen(file_name55, "w");
  //out56 = fopen(file_name56, "w");
  out57 = fopen(file_name57, "w");
  out58 = fopen(file_name58, "w");
  out59 = fopen(file_name59, "w");
  out60 = fopen(file_name60, "w");
  //out61 = fopen(file_name61, "w");
  out62 = fopen(file_name62, "w");
  out63 = fopen(file_name63, "w");
  out64 = fopen(file_name64, "w");
  out65 = fopen(file_name65, "w");
  out66 = fopen(file_name66, "w");
  out67 = fopen(file_name67, "w");
  out68 = fopen(file_name68, "w");
  out69 = fopen(file_name69, "w");
  out70 = fopen(file_name70, "w");
  out71 = fopen(file_name71, "w");
  out72 = fopen(file_name72, "w");
  out73 = fopen(file_name73, "w");
  out74 = fopen(file_name74, "w"); 
 


for(n_sim = 0; n_sim < TOT_SIM; n_sim++){
    t = 0;
    
    Initialization();
    
    Price[0]=0.3;
    t = 1;
    while(t < T){
      if(t < 10){
	Matrix_Random();
      }
      else{
      
      for(i = 0; i < N; i++){
        capacity[i]=(1.-haircut[i])*asset[i];
        if (capacity[i]<1.){
        capacity[i]=4.;
        }
        fprintf(out49,"%d  %f %d \n",t,capacity[i],n_sim );
     fflush(out49);
           }



equityMax=0.;
      for(i = 0; i < N; i++){
           if (equity[i]>equityMax){
             equityMax=equity[i];
             }
            }        
       


for(i = 0; i < N; i++){
        probfail[i]=(1-(equity[i]/equityMax));
           }


for(j = 0; j < N; j++){
   for(i = 0; i < N; i++){
       if(credit[j][i]==0.){
        
        a[j]=-(prud*asset[i])+(delta*asset[j]);
        //a[j]=(delta*asset[i])-(prud*asset[j]);
                
        b[j]=probfail[i]*(alpha*asset[i]-capacity[i]);

        c[j]=(1-probfail[i])*capacity[i];
         
        interest_interbank[j][i]=((a[j]-b[j])/c[j]);
        if (interest_interbank[j][i]<0.02){
           interest_interbank[j][i]=0.02;
             }
       // if (interest_interbank[j][i]>0.17){
         //  interest_interbank[j][i]=0.17;
           //  }
            }
      //fprintf(out39,"%d  %d %d %f %f %f  %f %d \n",t,j,i, interest_interbank[j][i], a[j],b[j], c[j],n_sim );
     //fflush(out39);
      }
     }



/*for(j = 0; j < N; j++){
   for(i = 0; i < N; i++){
         printf("%d  %d %d %f \n",t,j,i, interest_interbank[j][i] );
         fflush(out);          
         }
        }*/



   
     for(i = 0; i < N; i++){
         sommatasso[i]=0.;
         collegati[i]=0;
        }    



     for(j = 0; j < N; j++){ 
       for(i = 0; i < N; i++){
          
              if(credit[j][i]==0.){
               sommatasso[j]+=interest_interbank[j][i];
                collegati[j]+=1;
                tasso[j]=sommatasso[j]/ collegati[j];
               }
              }          
         }
            
	Matrix_Preferential();  
      }



     for(i = 0; i < N; i++){
       for(j = 0; j < N; j++){
          if(marketBank[i][j] == 1){
             fprintf(out57,"%d %d %d %d \n",i,j, t,n_sim);
             fflush(out57);    
          }
         }
        } 


         
   totfailures=0;

     /* for(j = 0; j < N; j++){
         printf("%d  %d %f %d %f\n",t,j,tasso[j],i_max2,tasso[i_max2]);
         fflush(out);
         }*/


       for(i = 0; i < N; i++){
      incominglink[i] =0;
            }
    
    for (i=0;i<N; i++){
      for(j=0; j<N; j++){
	if(marketBank[i][j]==1 && i!=j){     
	  incominglink[j]+=1;
	}
      }
    }
    
    for (i=0;i<N; i++){
      fprintf(out12,"%d %d %d %d\n",t, i, incominglink[i],n_sim);
      fflush(out12);     
    }
    
    
    probfaillinked=0.;
    probfailnonlinked=0.;
    num=0;
    num1=0;
    for (i=0;i<N; i++){
      if(vicino[i]==guru){
      probfaillinked+=probfail[i];
      num+=1;  
      probfaillinked=probfaillinked/num;
       }
      else{
       probfailnonlinked+=probfail[i];
       num1+=1;
       probfailnonlinked=probfailnonlinked/num1;
      } 
    }
    
    //fprintf(out47,"%d %f %d \n",t,  probfaillinked,n_sim);
    //  fflush(out47); 
      
     //  fprintf(out48,"%d %f %d \n",t, probfailnonlinked ,n_sim);
    //  fflush(out48); 
    
    

     incominglinkMax = -1000.;
  guru = 0;
  for(j = 0; j < N; j++){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru = j;
    }
  }
  
  incominglinkMax = -1000.;
  guru2 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru2 = j;
      }
    }
  }
 
 incominglinkMax = -1000.;
  guru3 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru3 = j;
      }
     }
    }
  }
  
  incominglinkMax = -1000.;
  guru4 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
        if (j!=guru3){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru4 = j;
      }
      }
      }
    }
  }
  
  incominglinkMax = -1000.;
  guru5 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
       if (j!=guru3){
         if (j!=guru4){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru5 = j;
      }
      }
      }
      }
    }
  } 
  
  incominglinkMax = -1000.;
  guru6 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
       if (j!=guru3){
         if (j!=guru4){
           if (j!=guru5){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru6 = j;
      }
      }
      }
      }
      }
    }
  } 
  
  incominglinkMax = -1000.;
  guru7 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
       if (j!=guru3){
         if (j!=guru4){
            if (j!=guru5){
               if (j!=guru6){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru7 = j;
      }
      }
      }
      }
      }
      }
    }
  } 
  
  incominglinkMax = -1000.;
  guru8 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
       if (j!=guru3){
         if (j!=guru4){
            if (j!=guru5){
               if (j!=guru6){
                if (j!=guru7){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru8 = j;
      }
      }
      }
      }
      }
      }
      }
    }
  } 
  
  incominglinkMax = -1000.;
  guru9 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
       if (j!=guru3){
         if (j!=guru4){
            if (j!=guru5){
               if (j!=guru6){
                if (j!=guru7){
                  if (j!=guru8){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru9 = j;
      }
      }
      }
      }
      }
      }
      }
      }
    }
  } 
  
  incominglinkMax = -1000.;
  guru10 = -1;
  for(j = 0; j < N; j++){
   if (j!=guru){
     if (j!=guru2){
       if (j!=guru3){
         if (j!=guru4){
            if (j!=guru5){
               if (j!=guru6){
                if (j!=guru7){
                  if (j!=guru8){
                    if (j!=guru9){
    if(incominglink[j] > incominglinkMax){
      incominglinkMax =incominglink[j];
      guru10 = j;
      }
      }
      }
      }
      }
      }
      }
      }
      }
    }
  } 
  
  
  
 //fprintf(out53,"%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",t,guru,guru2,guru3,guru4,guru5,guru6,guru7,guru8,guru9,guru10,incominglink[guru],incominglink[guru2],incominglink[guru3],incominglink[guru4],incominglink[guru5],incominglink[guru6],incominglink[guru7],incominglink[guru8],incominglink[guru9],incominglink[guru10],n_sim );
   //     fflush(out53);
         
         
  
  if(vicino[guru]==guru2 || vicino[guru]==guru3 ||vicino[guru]==guru4 || vicino[guru]==guru5||vicino[guru]==guru6 || vicino[guru]==guru7||vicino[guru]==guru8 || vicino[guru]==guru9 || vicino[guru]==guru10){
 
  }
  if(vicino[guru2]==guru || vicino[guru2]==guru3 ||vicino[guru2]==guru4 || vicino[guru2]==guru5||vicino[guru2]==guru6 || vicino[guru2]==guru7||vicino[guru2]==guru8 || vicino[guru2]==guru9 || vicino[guru2]==guru10){
  //printf("%d %d %d\n",t,vicino[guru2], n_sim );
  //  fflush(out);
  }
  if(vicino[guru3]==guru || vicino[guru3]==guru2 ||vicino[guru3]==guru4 || vicino[guru3]==guru5||vicino[guru3]==guru6 || vicino[guru3]==guru7||vicino[guru3]==guru8 || vicino[guru3]==guru9 || vicino[guru3]==guru10){
  //printf("%d %d %d\n",t,vicino[guru3], n_sim );
    //fflush(out);
  }
  if(vicino[guru4]==guru2 || vicino[guru4]==guru3 ||vicino[guru4]==guru || vicino[guru4]==guru5||vicino[guru4]==guru6 || vicino[guru4]==guru7||vicino[guru4]==guru8 || vicino[guru4]==guru9 || vicino[guru4]==guru10){
  //printf("%d %d %d\n",t,vicino[guru4], n_sim );
    //fflush(out);
  }
  if(vicino[guru5]==guru2 || vicino[guru5]==guru3 ||vicino[guru5]==guru4 || vicino[guru5]==guru||vicino[guru5]==guru6 || vicino[guru5]==guru7||vicino[guru5]==guru8 || vicino[guru5]==guru9 || vicino[guru5]==guru10){
  //printf("%d %d %d\n",t,vicino[guru5], n_sim );
    //fflush(out);
  }
  if(vicino[guru6]==guru2 || vicino[guru6]==guru3 ||vicino[guru6]==guru4 || vicino[guru6]==guru5||vicino[guru6]==guru || vicino[guru6]==guru7||vicino[guru6]==guru8 || vicino[guru6]==guru9 || vicino[guru6]==guru10){
  //printf("%d %d %d\n",t,vicino[guru6], n_sim );
    //fflush(out);
  }
  if(vicino[guru7]==guru2 || vicino[guru7]==guru3 ||vicino[guru7]==guru4 || vicino[guru7]==guru5||vicino[guru7]==guru6 || vicino[guru7]==guru||vicino[guru7]==guru8 || vicino[guru7]==guru9 || vicino[guru7]==guru10){
  //printf("%d %d %d\n",t,vicino[guru7], n_sim );
    //fflush(out);
  }
  if(vicino[guru8]==guru2 || vicino[guru8]==guru3 ||vicino[guru8]==guru4 || vicino[guru8]==guru5||vicino[guru8]==guru6 || vicino[guru8]==guru7||vicino[guru8]==guru || vicino[guru8]==guru9 || vicino[guru8]==guru10){
  //printf("%d %d %d\n",t,vicino[guru8], n_sim );
    //fflush(out);
  }
  if(vicino[guru9]==guru2 || vicino[guru9]==guru3 ||vicino[guru9]==guru4 || vicino[guru9]==guru5||vicino[guru9]==guru6 || vicino[guru9]==guru7||vicino[guru9]==guru8 || vicino[guru9]==guru || vicino[guru9]==guru10){
  //printf("%d %d %d\n",t,vicino[guru9], n_sim );
    //fflush(out);
  }
  if(vicino[guru10]==guru2 || vicino[guru10]==guru3 ||vicino[guru10]==guru4 || vicino[guru10]==guru5||vicino[guru10]==guru6 || vicino[guru10]==guru7||vicino[guru10]==guru8 || vicino[guru10]==guru9 || vicino[guru10]==guru){
  //printf("%d %d %d\n",t,vicino[guru10], n_sim );
   // fflush(out);
  }
  
  
 //NEW PART: CORE PERIPHERY 22/07/2014
 
 for(i = 0; i < N; i++){
 core[i]=0;
 periphery[i]=0;
 }
 
 coretot=0;
 peripherytot=0;
 inlinkcore=0;
 inlinkperi=0;
 //threshold=0.5*incominglink[guru];
 
 thre=incominglink[guru]+incominglink[guru2]+incominglink[guru3]+incominglink[guru4]+incominglink[guru5]+incominglink[guru6]+incominglink[guru7]+incominglink[guru8]+incominglink[guru9]+incominglink[guru10];
 thre1=(float)thre/10;
 threshold=0.5*thre1;
 for(i = 0; i < N; i++){
  if(incominglink[i]>=threshold){
  core[i]=1;
  coretot+=1;
  inlinkcore+=incominglink[i];
  } 
  else{
  periphery[i]=1;
  peripherytot+=1;
  inlinkperi+=incominglink[i];
  }
 }
 
 meanlinkcore=0.;
 meanlinkperi=0.;
 
 
 meanlinkcore=(float)inlinkcore/coretot;
 meanlinkperi=(float)inlinkperi/peripherytot;
 
    fprintf(out60,"%d %d %d %d\n",t,coretot,peripherytot, n_sim);
    fflush(out60);
    
    
    
    
    fprintf(out62,"%d %d %d %d \n",t,inlinkcore,inlinkperi, n_sim);
    fflush(out62);
    
    fprintf(out63,"%d %f %f %d\n",t,meanlinkcore,meanlinkperi, n_sim);
    fflush(out63);
    
  for(i = 0; i < N; i++){  
  meaninterestcore[i]=0.;
  meaninterestperi[i]=0.;
  }
  
  for(j = 0; j < N; j++){
    if(core[j]==1){
      for(i = 0; i < N; i++){
       meaninterestcore[j]+=interest_interbank[j][i];
       }
       }
     else{
     for(i = 0; i < N; i++){
       meaninterestperi[j]+=interest_interbank[j][i];
       }
     }
   } 
    
    for(j = 0; j < N; j++){
    if(core[j]==1){
      meaninterestcore[j]=(float)meaninterestcore[j]/100;
      }
     else{
      meaninterestperi[j]=(float)meaninterestperi[j]/100;
     }
    }
    
    
    meanintcore=0.;
    meanintperi=0.;
    for(j = 0; j < N; j++){
    if(core[j]==1){
    meanintcore+= meaninterestcore[j];
    }
    else{
    meanintperi+=meaninterestperi[j];
    }
    }
    
    meanintcore=(float)meanintcore/coretot;
    meanintperi=(float)meanintperi/peripherytot;
    
    fprintf(out64,"%d %f %f %d\n",t, meanintcore, meanintperi, n_sim);
    fflush(out64);
    
        meancapacitycore=0.;
        meancapacityperi=0.;
        totcapacitycore=0.;
        totcapacityperi=0.; 
    for(i = 0; i < N; i++){
      if(core[i]==1){
       totcapacitycore+=capacity[i];
        }
    else{
       totcapacityperi+=capacity[i];
        }
       }
      
      meancapacitycore=(float)totcapacitycore/coretot;
      meancapacityperi=(float)totcapacityperi/peripherytot;
    
    fprintf(out71,"%d %f %f %d\n",t, totcapacitycore, totcapacityperi, n_sim );
             fflush(out71);
        
        fprintf(out72,"%d %f %f %d\n",t, meancapacitycore, meancapacityperi, n_sim );
             fflush(out72);
    
    
double guruperc, enne, minchia,  minchia1,incominglinkperc;
guruperc=0.;
incominglinkperc=0.;
enne=0.;
minchia=0.;
minchia1=0.;
enne=N;
minchia=guru;
minchia1=incominglink[guru];
guruperc=(minchia/enne);
incominglinkperc=(minchia1/enne);

//fprintf(out40,"%d %d %d\n",t,guru, n_sim );
  //  fflush(out40);

 //fprintf(out28,"%d %lf %d\n",t,guruperc, n_sim );
   // fflush(out28);
 //fprintf(out30,"%d %lf %d\n",t, incominglinkperc, n_sim);
   // fflush(out30);
 //fprintf(out31,"%d %f %d\n",t,fit[guru] , n_sim);
   // fflush(out31);
    
   for (i=0;i<N; i++){
    fprintf(out8,"%d %d %f %d\n", t,i,equity[i], n_sim);
    fflush(out8);  
    }

   for (i=0;i<N; i++){
   fprintf(out21,"%d %d %f %d\n",t,i,liquidity[i], n_sim);
   fflush(out21);
   }
     

 
   // fprintf(out22,"%d %f %f %d \n",t,liquidity[guru], liquidityMax,n_sim); 
    // fflush(out22);  
     
     // fprintf(out50,"%d %f %f %f %f %f %f %f %f %f %d \n",t,liquidity[guru2],liquidity[guru3],liquidity[guru4],liquidity[guru5],liquidity[guru6],liquidity[guru7],liquidity[guru8],liquidity[guru9],liquidity[guru10],n_sim );
       // fflush(out50);

      fprintf(out1,"%d  %f %d \n",t,liquidity[i_max2],n_sim);
     fflush(out1);   

    fprintf(out2,"%d  %f %d\n",t,tasso[i_max2],n_sim);
     fflush(out2);  

    //fprintf(out33,"%d  %f %d\n",t,equity[guru],n_sim);
     //fflush(out33);  
   //fprintf(out34,"%d  %f %d \n",t,equity[i_max],n_sim);
    // fflush(out34);  
    //fprintf(out35,"%d  %f %d \n",t,equity[i_max2],n_sim);
     //fflush(out35);    
  
   



    //fprintf(out23,"%d  %f %d %d\n",t,tasso[guru], guru,n_sim);
    // fflush(out23);

    fprintf(out24,"%d  %f %d\n",t,tasso[i_max],n_sim);
    fflush(out24);

    fprintf(out25,"%d  %f %d\n",t,liquidity[i_max],n_sim);
     fflush(out25); 

     //fprintf(out26,"%d  %f %d\n",t,badDebt[guru],n_sim);
     //fflush(out26); 
     //fprintf(out36,"%d  %f %d\n",t,badDebt[i_max],n_sim);
     //fflush(out36);  
    //fprintf(out37,"%d  %f %d \n",t,badDebt[i_max2],n_sim);
    // fflush(out37);
     
      //fprintf(out52,"%d %f %f %f %f %f %f %f %f %f %d \n",t,tasso[guru2],tasso[guru3],tasso[guru4],tasso[guru5],tasso[guru6],tasso[guru7],tasso[guru8],tasso[guru9],tasso[guru10],n_sim );
        //fflush(out52);


    
       depositimedi=0.0;
       equitymedia=0.;
       baddebtmedi=0.;
       liquiditymedia=0.;
       assetmedio=0.;
      

          depositimedi=totdeposit-deposit[guru];
          depositimedi=depositimedi/(N-1);

          equitymedia=totequity-equity[guru];
          equitymedia=equitymedia/(N-1);

          baddebtmedi=totbaddebt-badDebt[guru];
          baddebtmedi=baddebtmedi/(N-1);

          liquiditymedia=totliquidity-liquidity[guru];
          liquiditymedia=liquiditymedia/(N-1);
          
          assetmedio=totasset-asset[guru];
          assetmedio=assetmedio/(N-1);
          
       

          

         fprintf(out,"%d %f %d \n", t,depositimedi, n_sim);
         fflush(out);

         fprintf(out19,"%d %f %f %d\n",t, equitymedia,equity[guru],n_sim);
         fflush(out19);

         fprintf(out27,"%d %f %d\n",t,baddebtmedi,n_sim);
         fflush(out27);
    
         fprintf(out29,"%d %f %d\n",t,liquiditymedia,n_sim);
         fflush(out29);
         
         fprintf(out46,"%d %f %d\n",t,assetmedio,n_sim);
         fflush(out46);
      
    

 /*  int n1;
  float aux;
  
    for(i=0;i<N;i++){
      rateorder[i] = i;
    }
  aux=0.;
  n1 = 0;
  
    for(i=0;i<N;i++){
      for(j=i+1;j<N;j++){
	if(probfail[i]<probfail[j]){
	  aux = probfail[i];
	  probfail[i] = probfail[j];
	  probfail[j] = aux;
	  n1 = rateorder[i];
	  rateorder[i] = rateorder[j];
	  rateorder[j] = n1;
	}
    
      }
    }   */


    totbaddebt=0.;
    granted=0.;


  
        //Price[t]=fabs(Price[t-1] *(1. + 0.01*gasdev(&idum) ));
        
        
        Price[t]=0.001;
        
   
     
      for (i=0;i<N; i++){
     richiesta[i]=0.;     
    }
    
      for (i=0;i<N; i++){
     concesso[i]=0.;     
    }

	 for(i = 0; i < N; i++){  
         shocked[i]=0;
         } 
     
     
     
    
     for(z = 0; z < S; z++){
       i=ran2(&idum)*N;
       if(shocked[i]!=1){
        shocked[i]=1;
        }
       else{
       while (shocked[i]==1)
          i=ran2(&idum)*N;
          shocked[i]=1;
       }
     }
     
     
    
    
     

	//for(i = 0; i <shokkati; i++){    questa va bene se si decide di shockare solo alcune banks
        for(i = 0; i < N; i++){
           
           if(shocked[i]==1) {
         
         new_deposit[i] =/*(deposit[i] *(1. + 0.035*gasdev(&idum) ))*/((ran2(&idum)*0.6)+0.65)*deposit[i];
         }
         else{
         new_deposit[i]=deposit[i];
        }
           
           
         /*if((t/tau[i])*tau[i] == t) {
         //if(i!=guru){
         new_deposit[i] =/*(deposit[i] *(1. + 0.035*gasdev(&idum) ))((ran2(&idum)*0.6)+0.65)*deposit[i];*/
        // }
         //else{
        // new_deposit[i]=deposit[i];
        // }
	  deltaD[i] = new_deposit[i] - deposit [i];  
	  liquidity[i] += deltaD[i];
	  deposit[i] = new_deposit[i];
          asset[i]  = loan[i] + liquidity[i];
          leverage[i] = loan[i] / equity[i];
            if (liquidity[i]< 0.0) {
               Dloan[i]= fabs(liquidity[i]);
               liquidity[i]=0.0;
               }
	    else {
               Dloan[i]= 0.0; 
               
              }
             // }
             /*else{
              new_deposit[guru]=((ran2(&idum)*0.6)+0.5)*deposit[guru];
	      deltaD[guru] = new_deposit[guru] - deposit [guru]; 
	      liquidity[guru] += deltaD[guru];
	      deposit[guru] = new_deposit[guru];
              asset[guru]  = loan[guru] + liquidity[guru];
              leverage[guru] = loan[guru] / equity[guru];
            if (liquidity[guru]< 0.0) {
               Dloan[guru]= fabs(liquidity[guru]);
               liquidity[guru]=0.0;
               }
	    else {
               Dloan[guru]= 0.0;  
              }
               }*/
           
             } 
             
             
                     
//calcolo l'asked:   


 asked=0.;
      for(i = 0; i < N; i++){
             asked += Dloan[i];
                }
     fprintf(out4,"%d %f %d\n", t,asked,n_sim);
          fflush(out4); 


           for(i = 0; i < N; i++){
           new_deposit[i]=0.;
            }
            
     /* for(i = 0; i < N; i++){   
 printf("%d %d %f \n",t,i,liquidity[i] );
        fflush(out);
         }*/
         
   
  //fprintf(out38,"%d  %f %d %f %f %d\n",t,deposit[guru], guru, tasso[guru], leverage[guru],n_sim);
    // fflush(out38);
     
     
   //  fprintf(out51,"%d %f %f %f %f %f %f %f %f %f %d \n",t,leverage[guru2],leverage[guru3],leverage[guru4],leverage[guru5],leverage[guru6],leverage[guru7],leverage[guru8],leverage[guru9],leverage[guru10],n_sim );
     //   fflush(out51);
  
        
             BanksPayBanks();
             
             //fprintf(out41,"%d %d %d %d \n",t,guru, failB[guru],n_sim );
             //fflush(out41);
             
           //  fprintf(out54,"%d %d %d %d %d %d %d %d %d %d %d \n",t,failB[guru2],failB[guru3],failB[guru4],failB[guru5],failB[guru6],failB[guru7],failB[guru8],failB[guru9],failB[guru10],n_sim );
        //fflush(out54);
        
        
        
        mortiguru2=0;
        mortiguru3=0;
        mortiguru4=0;
        mortiguru5=0;
        mortiguru6=0;
        mortiguru7=0;
        mortiguru8=0;
        mortiguru9=0;
        mortiguru10=0;
        
        meanmorticore=0.;
        meanmortiperi=0.;
        totmorticore=0.;
        totmortiperi=0.; 
    for(i = 0; i < N; i++){
     if(failB[i]==1){ 
      if(core[i]==1){
       totmorticore+=1;
        }
    else{
       totmortiperi+=1;
        }
       }
      }
      meanmorticore=(float)totmorticore/coretot;
      meanmortiperi=(float)totmortiperi/peripherytot;
    
    fprintf(out65,"%d %d %d %d \n",t, totmorticore, totmortiperi, n_sim );
             fflush(out65);
        
        fprintf(out66,"%d %f %f %d \n",t, meanmorticore, meanmortiperi, n_sim );
             fflush(out66);
             
             
        for(i = 0; i < N; i++){
          if(failB[i]==1){
            if (vicino[i]==guru2){
             mortiguru2++;
             }
             else if (vicino[i]==guru3){
             mortiguru3++;
             }
             else if (vicino[i]==guru4){
             mortiguru4++;
             }
             else if (vicino[i]==guru5){
             mortiguru5++;
             }
             else if (vicino[i]==guru6){
             mortiguru6++;
             }
             else if (vicino[i]==guru7){
             mortiguru7++;
             }
             else if (vicino[i]==guru8){
             mortiguru8++;
             }
             else if (vicino[i]==guru8){
             mortiguru9++;
             }
             else if(vicino[i]==guru10){
             mortiguru10++;
             }
            }
           }
          
       // fprintf(out56,"%d %d %d %d %d %d %d %d %d %d %d \n",t,mortiguru2,mortiguru3,mortiguru4,mortiguru5,mortiguru6,mortiguru7,mortiguru8,mortiguru9,mortiguru10,n_sim );
       // fflush(out56);
        
          
        meanbaddebtcore=0.;
        meanbaddebtperi=0.;
        totbaddebtcore=0.;
        totbaddebtperi=0.; 
    for(i = 0; i < N; i++){
      if(core[i]==1){
       totbaddebtcore+=badDebt[i];
        }
    else{
       totbaddebtperi+=badDebt[i];
        }
       }
      
      meanbaddebtcore=(float)totbaddebtcore/coretot;
      meanbaddebtperi=(float)totbaddebtperi/peripherytot;
    
    fprintf(out67,"%d %f %f %d \n",t, totbaddebtcore, totbaddebtperi, n_sim );
             fflush(out67);
        
        fprintf(out68,"%d %f %f %d \n",t, meanbaddebtcore, meanbaddebtperi, n_sim );
             fflush(out68);
             
         totbaddebt=0.;
             
         for(i = 0; i < N; i++){
          totbaddebt += badDebt[i];
           }
           
       fprintf(out18,"%d %f %d \n", t, totbaddebt,n_sim );
       fflush(out18);


    
             NewBanks();


          for(i = 0; i < N; i++){
              badDebt[i] = 0.;
            }

            
         


        for(i = 0; i < N; i++){
             for(j = 0; j < N; j++){
              executed[i]=0;
             }
            }


      crunchTot=0.0;
      for(i = 0; i < N; i++){
      
              if (Dloan[i]>0.0  && connected[i] == 0){
                 firesale[i]= Dloan[i]/Price[t];
                 loanintero[i]=loan[i];
                 loan[i] -= firesale[i];  
                 if (loan[i]>=0.0){
                 Dloan[i]=0.0;
                 crunch[i]=Dloan[i];
                 equity[i]-=(1-Price[t])*firesale[i];
                 new_deposit[i]=0.0;
                 deltaD[i]=0.0;
                 firesale[i]=0.0;
                 interbankloan[i]=0.0;
                 interbankdebt[i]=0.;
                 failB[i]=0;
                 asset[i]=loan[i]+liquidity[i];
                 leverage[i]=loan[i]/equity[i];
                 }
                 else{
                 Dloan[i]-=Price[t]*loanintero[i];
                 crunch[i]= Dloan[i];
                 failB[i]=1;
                 Dloan[i]=0.0;
                 equity[i]=0.0;
                 }
                }
                crunchTot+= crunch[i];
                }




  
      Trade();
      
      for(i = 0; i < N; i++){
                 intradayleverage[i]=0.;
                 }

              for(i = 0; i < N; i++){
                 intradayleverage[i]=(loan[i]+interbankloan[i])/equity[i];
                 fprintf(out59,"%d %d %f %d\n", t,i,intradayleverage[i], n_sim);
          fflush(out59);

                 }

        meanlevacore=0.;
        meanlevaperi=0.;
        totlevacore=0.;
        totlevaperi=0.; 
    for(i = 0; i < N; i++){
      if(core[i]==1){
       totlevacore+=intradayleverage[i];
        }
    else{
       totlevaperi+=intradayleverage[i];
        }
       }
      
      meanlevacore=(float)totlevacore/coretot;
      meanlevaperi=(float)totlevaperi/peripherytot;
    
    fprintf(out69,"%d %f %f %d \n",t, totlevacore, totlevaperi, n_sim );
             fflush(out69);
        
        fprintf(out70,"%d %f %f %d \n",t, meanlevacore, meanlevaperi, n_sim );
             fflush(out70);


 
//calcolo il granted:

      granted=0.;
      for(i = 0; i < N; i++){
             for(j = 0; j < N; j++){
                granted += credit[j][i];
                }
               }
        fprintf(out5,"%d %f %f %d\n", t,granted, asked-granted, n_sim);
          fflush(out5);


      loanToRate=0.;
      transaction=0.;
      MeanRate[t]=0.;
      MeanRate[1]=0.04;
        for(j = 0; j < N; j++){
          for(i = 0; i < N; i++){
                if (credit[j][i]!=0.){
                loanToRate += credit[j][i]*interest_interbank[j][i];
                transaction += credit[j][i];
                // printf("%d  %d %d %f %f %d\n",t,j,i, credit[j][i], interest_interbank[j][i], guru );
                // fflush(out); 
                   }
                }
               }
          
         if(transaction!=0.){
         MeanRate[t]=loanToRate/transaction;
            }
        else{
         MeanRate[t]=MeanRate[t-1];
          }
        fprintf(out6,"%d %f %f %d \n", t, MeanRate[t],transaction,n_sim);
       fflush(out6);





      Firesale();
      
      
    
      /*NetworkStatistics();*/

  
      
      for(i = 0; i < N; i++){
      totfailures += failB[i];
      }
      fprintf(out7,"%d %d %d\n",t, totfailures,n_sim);
      fflush(out7);
	
   //    AggregateStatistics();

     //controllo fallimenti per isolamento:
           no_connect_fail=0;
           for(i = 0; i < N; i++){
              if (connected[i]==0){
            no_connect_fail += failB[i];
                }
               }
     
//fprintf(out26,"%d  %f %d\n",t,badDebt[guru],n_sim);
  //   fflush(out26); 
  

      NewBanks();

      AggregateStatistics();

//ora calcolo il leverage massimo cheuserò per calcolare gli haircuts:
      maxleverage=0.;
      for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
          if (matched[i][j]==0){ 
           if (leverage[i]>maxleverage){
             maxleverage=leverage[i];
             }
            }        
           }
          }
       


  
       for(i = 0; i < N; i++){
           haircut[i]=leverage[i]/maxleverage; 
         }


//ora calcolo il tasso di interesse che chiederenno dal giorno successivo:

     
   
     /* for(i = 0; i < N; i++){
      fprintf(out10,"%d %d %f %f %f %f %f %f %f %f %d\n",t, i, loan[i], liquidity[i], deposit[i],equity[i], liquidity[i]+ loan[i]+interbankloan[i], equity[i]+deposit[i]+interbankdebt[i],interbankloan[i],interbankdebt[i],n_sim);
      fflush(out10);
      } 
     
     for(i = 0; i < N; i++){
       c[i]=liquidity[i]+ loan[i]+interbankloan[i];
       d[i]= equity[i]+deposit[i]+interbankdebt[i];
          if(c[i]!=d[i]){
           e[i]=1;
           }
        else{ 
           e[i]=0;
          }*/
        

      t++;
    }
  }
  return 0;
  
}

/**************END MAIN***************/

/*************FUNCTIONS***************/

///---------------------------------///
void Initialization(){
  for(i = 0; i < N; i++){
    loan[i]          = 120.;
    liquidity[i]     = 30.;
    deposit[i]       = 135.;
    equity[i]        = 15.;
    new_deposit[i]   = 0.;
    deltaD[i]        = 0.;
    Dloan[i]         = 0.;
    firesale[i]      = 0.;
    crunch[i]        = 0.;
    interbankloan[i] = 0.;
    interbankdebt[i] = 0.;
    badDebt[i]       = 0.;
    rationed[i]      = 0.;
    failB[i]         = 0.;
    asset[i]         = loan[i] + liquidity[i];
    leverage[i]      = loan[i] / equity[i];
    haircut[i]       =0.3;
    a[i]             =0.;
    b[i]             =0.;
    c[i]             =0.;
    }


    


     for(i = 0; i < N; i++){
     for(j = 0; j < N; j++){
     matched[i][j]=0;
       }
      }


        equityMax=0.;
      for(i = 0; i < N; i++){
           if (equity[i]>equityMax){
             equityMax=equity[i];
             }
            }        
       
       for(i = 0; i < N; i++){
        probfail[i]=0.2;
           }
       

        for(i = 0; i < N; i++){
        capacity[i]=(1.-haircut[i])*asset[i];
           }


   for(i = 0; i < N; i++){
     for(j = 0; j < N; j++){
     interest_interbank[j][i]=0.04;
  
    }
   }
  }
///---------------------------------///

///---------------------------------///
int marked[N];
int mark = 0;
void Matrix_Random(){
  int i , j ;
  for ( i = 0; i < N; i ++ ){
	connected[i] = 0;
	for (  j = 0; j < N; j ++ ){
	  marketBank[i][j] = 0;
	}
  }
  srand((unsigned)time(0));
  int ne = 0;
  for(i = 0; i < N; i++){
	if(connected[i] != 1){
	  if(ran2(&idum) < RANDOM_CONNECTIVITY){
	j = ran2(&idum)*N;
	if(i != j ){
	  marketBank[i][j] = 1;
	  vicino[i] = j;
	  connected[i] = 1;
	  ne++;
	}
	else{
// 	  printf("CHE SUCCEDE!  %d %d %d \n",t,i,j);
	  while(i == j)
		j = ran2(&idum)*N;
	  marketBank[i][j] = 1;
	  vicino[i] = j;
	  connected[i] = 1;
// 	  printf("VEDIAMO Adesso  %d %d %d \n",t,i,j);
	}
	  }
	  else{
	// printf("RAZIONAMENTO %d %d \n",t,i);
	  }
	}
  }
  /*    marketBank[1][7] = 0;     //questo serve per il debug
	  marketBank[2][0] = 0;
	  marketBank[6][3] = 0;
	  marketBank[9][0] = 0;
	  marketBank[4][8] = 0;  
	  
	  connected[0] = 1;
	  connected[2] = 0;  */
	  

	
  
	  for ( i = 0; i < N; i ++ ){
	marked[i] = -1;
	  }
	  for ( i = 0; i < N; i ++ ){
	if ( marked[i] == -1 ){
	  mark ++;
	  dfs( i );
	}
	  }
}

void dfs( int n ) {
  if ( marked[n] != -1 ){
    return;
  }
  marked[n] = mark;
  int i;
  for ( i = 0; i < N; i ++ ){
    if ( (marketBank[n][i] == 1 || marketBank[i][n]==1 ) && marked[i] == -1)
      dfs( i );
  }
}
///---------------------------------///
void Matrix_Preferential(){
  float beta[N],interestMin;
  int link[N];
  int k =0;  int switching = 0;
  float ref = 0.;
  float fact = 0.;
  float prob = 0.;
  float fitness = 0.;
  
  liquidityMax = -1000.;
  i_max = -1;
  for(j = 0; j < N; j++){
    if(liquidity[j] > liquidityMax){
      liquidityMax =liquidity[j];
      i_max = j;
    }
  }

   /*equityMax = -1000.;
  i_max = -1;
  for(j = 0; j < N; j++){
    if(equity[j] > equityMax){
      equityMax =equity[j];
      i_max = j;
    }
  }*/




  interestMin = 1000.;
  i_max2 = -1;
  for(j = 0; j < N; j++){
    if(tasso[j] < interestMin){
      interestMin =tasso[j];
      i_max2 = j;
    }
  }






  for(i = 0; i < N; i++){
    fit[i] = 0.;
    beta[i] = 0.;
    outgoinglink[i]=0;
    incominglink[i]=0;
    link[i]=0;
  }
  for(j = 0; j < N; j++){
    fit[j] = (double)(ypsilon*(liquidity[j]/liquidityMax)+(1.-ypsilon)*(interestMin/tasso[j]));
    //fit[j] = (double)(ypsilon*(equity[j]/equityMax)+(1.-ypsilon)*(interestMin/tasso[j]));
     fprintf(out20,"%d  %f %d \n",t,fit[j],n_sim);
     fflush(out20);
       }

    


 for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      marketBank[i][j] = 0;
	  marketBank[j][i] = 0;
    }
  }
  for(i = 0; i < N; i++){
    beta[i] = /*(ran2(&idum)*T_inv)+5;*/T_inv;  ///temperatura (inverso) di i  CHECK!
  }
 for(i = 0; i < N; i++){
    j = i;  
    while(j == i){
      j = ran2(&idum)*N;
    }
    k = vicino[i];
    prob = ran2(&idum);
    fitness = ((fit[j])-fit[k]);  ///aggiungere i costi di transazione???-trans_cost
    ref = beta[i]*fitness;
    fact = (float)1./(1.+ exp(-ref));
    
    if( prob < fact && i != j){//fitness>0 // prob < fact
      switching++;
      marketBank[i][k] = 0;
      marketBank[i][j] = 1;
      vicino[i] = j;
      
    }
    else{
      marketBank[i][j] = 0;
      marketBank[i][k] = 1;
      vicino[i] = k; 
    } 
  }

for(i = 0; i < N; i++){
    connected[i] = 1;
  }


double cazzo, i_maxperc, i_max2perc, nerd, nerd1;

i_maxperc=0.;
i_max2perc=0.;
cazzo=0.;
cazzo=N;
nerd=i_max;
nerd1=i_max2;
i_maxperc=(nerd/cazzo);
i_max2perc=(nerd1/cazzo);




 fprintf(out13,"%d %lf %d\n",t,i_maxperc,n_sim);
   fflush(out13);

fprintf(out3,"%d  %lf %d \n", t,i_max2perc,n_sim);
          fflush(out3);


}




///---------------------------------///
float gasdev(long *idum)
{
        float ran2(long *idum);
        static int iset=0;
        static float gset;
        float fac,rsq,v1,v2;
 
        if  (iset == 0) {
                do {
                        v1=2.0*ran2(idum)-1.0;
                        v2=2.0*ran2(idum)-1.0;
                        rsq=v1*v1+v2*v2;
                } while (rsq >= 1.0 || rsq == 0.0);
                fac=sqrt(-2.0*log(rsq)/rsq);
                gset=v1*fac;
                iset=1;
                return v2*fac;
        } else {
                iset=0;
                return gset;
        }
}


///---------------------------------///
void decreasing(){
  int n1;
  float aux;
  
    for(i=0;i<N;i++){
      rateorder[i] = i;
    }
  
  n1 = 0;
  
    for(i=0;i<N;i++){
      for(j=i+1;j<N;j++){
	if(probfail[i]<probfail[j]){
	  aux = probfail[i];
	  probfail[i] = probfail[j];
	  probfail[j] = aux;
	  n1 = rateorder[i];
	  rateorder[i] = rateorder[j];
	  rateorder[j] = n1;
	}
 /*  printf("%d %f %f %d %d\n",t,aux, probfail[i],i, rateorder[i] );
    fflush(out);*/
      }
    }
  
}
///---------------------------------///

void Trade(){
int totinside;
totinside=0;

 for(i = 0; i < N; i++){
 insideinter[i]=0;
}
   
  for(i = 0; i < N; i++){
      for(j = 0; j < N; j++){
          if (Dloan[i]>0.  && marketBank[i][j] == 1 && liquidity[j] > 0.0){
             insideinter[i]=1;
             matched[i][j]=1;
             if (Dloan[i] < liquidity[j]){
                liquidity[j]-= Dloan[i];
                interbankloan[j]+=Dloan[i];
                interbankdebt[i]=Dloan[i];
                credit[j][i]=Dloan[i];
                concesso[i]=Dloan[i];
                Dloan[i]=0.0;
                }
                else {
                interbankloan[j]+=liquidity[j];
                interbankdebt[i]=liquidity[j];
                credit[j][i]=liquidity[j];
                concesso[i]=liquidity[j];
                Dloan[i]-=liquidity[j];
                liquidity[j]=0.0;
                 }
                }
               }
              }
              
              
        for(i = 0; i < N; i++){
          for(j = 0; j < N; j++){
            if( marketBank[i][j] ==1 && credit[j][i] != 0.){
               fprintf(out58,"%d %d %d %d \n",i,j, t,n_sim);
               fflush(out58);    
              }
             }
            }  


       for(i = 0; i < N; i++){
          if(insideinter[i]==1){
            totinside+=1;
            }
            }
       fprintf(out11,"%d %d %d \n",t, totinside, n_sim);
       fflush(out11);

             } 
///---------------------------------///

void Firesale(){                                 
      /*for(i = 0; i < N; i++){
       for(j = 0; j < N; j++){
            if (Dloan[i]>0.0 ){
              firesale[i]=(Dloan[i])/ Price[t];
              loanintero[i]=loan[i];
              loan[i]-=firesale[i];
               if (loan[i]>= 0.0){
                   Dloan[i]=0.0;
                   crunch[i]=Dloan[i];
                   equity[i]-=(1-Price[t])*firesale[i];
                   if (equity[i]<= 0.0){
                      matched[i][j]=0;
                      equity[j]-=credit[j][i];
                      interbankloan[j]-=credit[j][i];
                      interbankdebt[i]=0.;
                      credit[j][i]=0.;
                      failB[i]=1;
                      executed[i]=0;
                      executed[j]=0;
                       } 
                      }     
               else{
               Dloanintera[i]=Dloan[i];
               Dloan[i]-=Price[t]*loanintero[i];
               crunch[i]=Dloan[i];
               Dloan[i]=0.;
               matched[i][j]=0;
               equity[j]-=credit[j][i];
               interbankloan[j]-=credit[j][i];
               interbankdebt[i]=0.;
               credit[j][i]=0.;
               failB[i]=1;
               executed[i]=0;
               executed[j]=0;        
                      }
                     }
              
             crunchTot+=crunch[i];
              }
              }*/


        for(i = 0; i < N; i++){
            if (Dloan[i]>0.0 ){
              
              rationed[i]=1;
              firesale[i]=(Dloan[i])/ Price[t];
              loanintero[i]=loan[i];
              loan[i]-=firesale[i];
               if (loan[i]>= 0.0){
                   Dloan[i]=0.0;
                   crunch[i]=Dloan[i];
                   equity[i]-=(1-Price[t])*firesale[i];
                   if (equity[i]<= 0.0){
                      interbankdebt[i]=0.;
                      //credit[j][i]=0.;
                      failB[i]=1;
                      executed[i]=0;
                       } 
                      }     
               else{
               Dloanintera[i]=Dloan[i];
               Dloan[i]-=Price[t]*loanintero[i];
               crunch[i]=Dloan[i];
               Dloan[i]=0.;
               interbankdebt[i]=0.;
               failB[i]=1;
               executed[i]=0;
                      }
                     }
                   }
                   
             totrationed=0;     
                for(i = 0; i < N; i++){    
                totrationed+=rationed[i];  
                  } 
                  
             fprintf(out43,"%d %d %d \n",t, totrationed, n_sim);
             fflush(out43);
              
             
             
            // fprintf(out55,"%d %d %d %d %d %d %d %d %d %d %d \n",t,rationed[guru2],rationed[guru3],rationed[guru4],rationed[guru5],rationed[guru6],rationed[guru7],rationed[guru8],rationed[guru9],rationed[guru10],n_sim );
      //  fflush(out55);
             
             
      totrichesto=0.;
      totconcesso=0;
      for(i = 0; i < N; i++){
         if (rationed[i]==1){
             totrichesto += richiesta[i];
             totconcesso +=concesso[i];
                }
               }
     fprintf(out44,"%d %f %f %f %d\n", t,totrichesto,totconcesso,totrichesto-totconcesso,  n_sim);
          fflush(out44); 

        meanrationcore=0.;
        meanrationperi=0.;
        totrationcore=0.;
        totrationperi=0.; 
    for(i = 0; i < N; i++){
    if (rationed[i]==1){
      if(core[i]==1){
       totrationcore+=1;
        }
    else{
       totrationperi+=1;
        }
       }
      }
      meanrationcore=(float)totrationcore/coretot;
      meanrationperi=(float)totrationperi/peripherytot;
    
    fprintf(out73,"%d %f %f %d \n",t, totrationcore, totrationperi, n_sim );
             fflush(out73);
        
        fprintf(out74,"%d %f %f %d \n",t, meanrationcore, meanrationperi , n_sim);
             fflush(out74);
             
             
             
                              
            
            for(i = 0; i < N; i++){       
              rationed[i]      = 0.;   
                 }  
                   

         for(i = 0; i < N; i++){
          for(j = 0; j < N; j++){
           if (failB[i]==1){ 
               loan[i]=0.;
               matched[i][j]=0;
               equity[j]-=credit[j][i];
               interbankloan[j]-=credit[j][i];
               badDebt[j]+=(credit[j][i]*(1.+interest_interbank[j][i]));
               interbankdebt[i]=0.;
               credit[j][i]=0.;
               failB[i]=1;
               executed[j]=0;
              } 
             }
            }
              
             }
            
///---------------------------------///

void BanksPayBanks(){

    for(i = 0; i < N; i++){
       for(j = 0; j < N; j++){
           if (credit[j][i] ){
          executed[i]=1;
          executed[j]=1;
            if (Dloan[i]==0.){
             if (liquidity[i] >= credit[j][i]*(1.+interest_interbank[j][i])){
              liquidity[i]-=credit[j][i]*(1.+interest_interbank[j][i]);
              equity[i]-= credit[j][i]*interest_interbank[j][i];
              new_deposit[i]=0.0;
              deltaD[i]=0.0;
              firesale[i]=0.0;
              interbankloan[i]=0.0;
              failB[i]=0;
              asset[i]=loan[i]+liquidity[i];
              leverage[i]=loan[i]/equity[i];
              liquidity[j]+=credit[j][i]*(1+interest_interbank[j][i]);
              equity[j]+=credit[j][i]*interest_interbank[j][i];
              new_deposit[j]=0.0;
              deltaD[j]=0.0;
              firesale[j]=0.0;
              interbankloan[j]-=credit[j][i];
              interbankdebt[i]=0.;
              failB[j]=0;
              asset[j]=loan[j]+liquidity[j];
              leverage[j]=loan[j]/equity[j];
              credit[j][i]=0.0;
              matched[i][j]=0;
               }
              else {
              residualcredit[j][i]=(credit[j][i]*(1.+interest_interbank[j][i]))-liquidity[i];
              liquidity[i]=0.;
              loanintero[i]=loan[i];
              firesale[i]=residualcredit[j][i]/Price[t];
              loan[i] -=firesale[i];
    
                       if (loan[i] >= 0.){
                       residualcredit[j][i]=0.;
                       equity[i]-=(1.-Price[t])*firesale[i]+(credit[j][i]*interest_interbank[j][i]);
                       liquidity[j]+=credit[j][i]*(1.+interest_interbank[j][i]);
                       equity[j]+=credit[j][i]*interest_interbank[j][i];
                       new_deposit[i]=0.0;
                       deltaD[i]=0.0;
                       firesale[i]=0.0;
                       interbankloan[i]=0.0;
                       asset[i]=loan[i]+liquidity[i];
                       leverage[i]=loan[i]/equity[i];
                       new_deposit[j]=0.0;
                       deltaD[j]=0.0;
                       firesale[j]=0.0;
                       interbankloan[j]-=credit[j][i];
                       interbankdebt[i]=0.;
                       failB[j]=0;
                       asset[j]=loan[j]+liquidity[j];
                       leverage[j]=loan[j]/equity[j];
                       credit[j][i]=0.;
                       matched[i][j]=0;
                       }
                       else{
                       residualcreditintero[j][i]=residualcredit[j][i];   
                       residualcredit[j][i]-=Price[t]*loanintero[i];
                       failB[i]=1;
                       liquidity[j]+=Price[t]*loanintero[i];
                       equity[j]-=credit[j][i]-(Price[t]*loanintero[i]);
                       badDebt[j]+=(credit[j][i]*(1.+interest_interbank[j][i]));//-(Price[t]*loanintero[i]);
                       leverage[j]=loan[j]/equity[j];
                       new_deposit[i]=0.0;
                       deltaD[i]=0.0;
                       firesale[i]=0.0;
                       interbankloan[i]=0.0;
                       asset[i]=loan[i]+liquidity[i];
                       leverage[i]=loan[i]/equity[i];
                       new_deposit[j]=0.0;
                       deltaD[j]=0.0;
                       firesale[j]=0.0;
                       interbankloan[j]-=Price[t]*loanintero[i];
                       interbankdebt[i]=0.;
                       asset[j]=loan[j]+liquidity[j];
                       leverage[j]=loan[j]/equity[j];
                       credit[j][i]=0.0;
                       matched[i][j]=0;
                       }
                      }
                     }
                  else{
                   loanintero[i]=loan[i];
                   firesale[i]=(Dloan[i]+(credit[j][i]*(1.+interest_interbank[j][i])))/Price[t];
                   loan[i] -=firesale[i];
                      if(loan[i]>=0.){
                         Dloan[i]=0.;
                         liquidity[j]+=(credit[j][i]*(1.+interest_interbank[j][i]));
                         equity[j]+=credit[j][i]*interest_interbank[j][i];
                         equity[i]-=(1.-Price[t])*firesale[i]+(credit[j][i]*interest_interbank[j][i]);
                         leverage[j]=loan[j]/equity[j];
                         new_deposit[i]=0.0;
                         deltaD[i]=0.0;
                         firesale[i]=0.0;
                         interbankloan[i]=0.0;
                         asset[i]=loan[i]+liquidity[i];
                         leverage[i]=loan[i]/equity[i];
                         new_deposit[j]=0.0;
                         deltaD[j]=0.0;
                         firesale[j]=0.0;
                         interbankloan[j]=0.0;
                         interbankdebt[i]=0.;
                         asset[j]=loan[j]+liquidity[j];
                         leverage[j]=loan[j]/equity[j];
                         credit[j][i]=0.;
                         failB[j]=0;
                         matched[i][j]=0;
                         }
                       else{
                         failB[i]=1;
                         firesale[i]=Price[t]*loanintero[i];
                         firesale[i]-=Dloan[i];
                            if(firesale[i]>0.){
                              if(firesale[i]>credit[j][i]){
                                if(firesale[i]>(credit[j][i]*(1.+interest_interbank[j][i]))){
                                liquidity[j]+=(credit[j][i]*(1.+interest_interbank[j][i]));
                                equity[j]+=credit[j][i]*interest_interbank[j][i];
                                
                                credit[j][i]=0.0;
                                new_deposit[i]=0.0;
                                deltaD[i]=0.0;
                                firesale[i]=0.0;
                                interbankloan[i]=0.0;
                                asset[i]=loan[i]+liquidity[i];
                                leverage[i]=loan[i]/equity[i];
                                new_deposit[j]=0.0;
                                deltaD[j]=0.0;
                                firesale[j]=0.0;
                                interbankloan[j]=0.0;
                                interbankdebt[i]=0.;
                                asset[j]=loan[j]+liquidity[j];
                                leverage[j]=loan[j]/equity[j];
                                matched[i][j]=0;
                                }
                                else{
                                liquidity[j]+=firesale[i];
                                equity[j]+=firesale[i]-credit[j][i];
                                badDebt[j]+=(credit[j][i]*(1.+interest_interbank[j][i]));//-firesale[i];
                                credit[j][i]-(Price[t]*loanintero[i]);
                                leverage[j]=loan[j]/equity[j];
                                credit[j][i]=0.0;
                                new_deposit[i]=0.0;
                                deltaD[i]=0.0;
                                firesale[i]=0.0;
                                interbankloan[i]=0.0;
                                asset[i]=loan[i]+liquidity[i];
                                leverage[i]=loan[i]/equity[i];
                                new_deposit[j]=0.0;
                                deltaD[j]=0.0;
                                firesale[j]=0.0;
                                interbankloan[j]=0.0;
                                interbankdebt[i]=0.;
                                asset[j]=loan[j]+liquidity[j];
                                leverage[j]=loan[j]/equity[j];
                                matched[i][j]=0;
                                }
                                }
                               else{
                               liquidity[j]+=firesale[i];
                               equity[j]-=credit[j][i]-firesale[i];
                               badDebt[j]+=(credit[j][i]*(1.+interest_interbank[j][i]));//-firesale[i];
                               leverage[j]=loan[j]/equity[j];
                               credit[j][i]=0.0;
                               new_deposit[i]=0.0;
                               deltaD[i]=0.0;
                               firesale[i]=0.0;
                               interbankloan[i]=0.0;
                               asset[i]=loan[i]+liquidity[i];
                               leverage[i]=loan[i]/equity[i];
                               new_deposit[j]=0.0;
                               deltaD[j]=0.0;
                               firesale[j]=0.0;
                               interbankloan[j]=0.0;
                               interbankdebt[i]=0.;
                               asset[j]=loan[j]+liquidity[j];
                               leverage[j]=loan[j]/equity[j];
                               matched[i][j]=0;
                                }
                               }
                             else{
                             badDebt[j]+=(credit[j][i]*(1.+interest_interbank[j][i]));
                             equity[j]-=credit[j][i];
                             leverage[j]=loan[j]/equity[j];
                             credit[j][i]=0.0;
                             new_deposit[i]=0.0;
                             deltaD[i]=0.0;
                             firesale[i]=0.0;
                             interbankloan[i]=0.0;
                             asset[i]=loan[i]+liquidity[i];
                             leverage[i]=loan[i]/equity[i];
                             new_deposit[j]=0.0;
                             deltaD[j]=0.0;
                             firesale[j]=0.0;
                             interbankloan[j]=0.0;
                             interbankdebt[i]=0.;
                             asset[j]=loan[j]+liquidity[j];
                             leverage[j]=loan[j]/equity[j];
                             matched[i][j]=0;
                              }
                             }
                            }  

                           } 
           } 
          } 

  for(i = 0; i < N; i++){
                 interbankloan[i]=0.;
                 interbankdebt[i]=0.;
                 insideinter[i]=0;  
                        }


   for(i = 0; i < N; i++){
               adequacy[i]=equity[i]/loan[i];
                 }

              for(i = 0; i < N; i++){
          if (adequacy[i]<=0.08 ){
             failB[i]=1;
              }
             }
         
        

       for(i = 0; i < N; i++){
          if (equity[i]<=0.0){
             failB[i]=1;
              }
             }

 
      for(i = 0; i < N; i++){
      totfailures += failB[i];
      }
     


      si_connect_fail=0;
           for(i = 0; i < N; i++){
              if (connected[i]==1){
            si_connect_fail += failB[i];
                }
               }
   //  fprintf(out9,"%d %d %d\n", t, si_connect_fail,n_sim );
    //      fflush(out9);  
     

      for(i = 0; i < N; i++){
         insideinter[i]=0;
        }
        
        
	  fallimento_linked=0;
	  fallimento_nolinked=0;
	  chi=0;
	  nochi=0;
    for(i = 0; i<N; i++){
	   if(i != guru && vicino[i]==guru){
	   chi++;
	    fallimento_linked+=failB[i];
	   }
      if(i != guru && vicino[i]!=guru){
	   nochi++;
	   fallimento_nolinked+=failB[i];
	   }
	}
	
	  // fprintf(out42,"%d %d %d %d\n", t, fallimento_linked, fallimento_nolinked,n_sim);
          // fflush(out42);
           
        
         }
 ///---------------------------------///
	void NewBanks(){

             

            for(i = 0; i < N; i++){
          if (equity[i]<=0.0 || loan[i]<0.){
             failB[i]=1;
              }
             }


          for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
              if(failB[i]==1 || failB[j]==1){
               credit[j][i]=0.0;
                 }
                }
               }
       

	  for(i = 0; i < N; i++){
	    if(failB[i] == 1){    
	      loan[i]          = 120;
              liquidity[i]     = 30;
              deposit[i]       = 135;
              equity[i]        = 15;
              new_deposit[i]   = 0.;
              deltaD[i]        = 0.;
              Dloan[i]         = 0.;
              firesale[i]      = 0.;
              crunch[i]        = 0.;
              rationed[i]      = 0 ;
              badDebt[i]       = 0.;
              interbankloan[i] = 0.;
              interbankdebt[i] = 0.;
              failB[i]         = 0.;
              asset[i]         = loan[i] + liquidity[i];
              leverage[i]      = loan[i] / equity[i];

	    }
	   }
          } 
 ///---------------------------------///
   void AggregateStatistics(){
      
      

          totloan=0.;
          totliquidity=0.;
          totdeposit=0.;
          totequity=0.;
          totnewdeposit=0.;
          totasset=0.;

          for(i = 0; i < N; i++){
          totloan += loan[i];
          totliquidity += liquidity[i];
          totdeposit += deposit[i];
          totequity += equity[i];
          totnewdeposit+= new_deposit[i];
          totasset+=asset[i];
           }




       fprintf(out14,"%d %f %d \n", t, totloan,n_sim);
       fflush(out14);
       fprintf(out15,"%d %f %d \n", t, totliquidity,n_sim);
       fflush(out15);
       fprintf(out16,"%d %f %d \n", t, totdeposit,n_sim);
       fflush(out16);
       fprintf(out17,"%d %f %d \n", t, totequity,n_sim);
       fflush(out17);
       fprintf(out45,"%d %f %d \n", t, totasset,n_sim);
       fflush(out45);
       
       
      // fprintf(out32,"%d %f %d\n", t, totnewdeposit,n_sim);
      // fflush(out32);
        }
              


 ///---------------------------------///
 /*************END FUNCTIONS***********/
 /************END OF CODE**************/
