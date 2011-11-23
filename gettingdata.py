#!/usr/bin/env python
from Database_v1 import *
import numpy as np

# additional packages required for modules:
# sqlalchemy psycopg2 cvxopt (pylab matplotlib scipy etc...)

# Get unnormalized data for regions
t, L_reg, GW_reg, GS_reg, datetime_offset, datalabels_reg = get_data_regions()

# In the making of the following timeseries, we keep load unnormalized and wind and solar resources normalized to 1
# The numbers of the rows that are being extracted are defined by the database's documentation, "Siemens_Abschluss_3_12.pdf" 
# and correspond to the respective region.

LoadDkEast=L_reg[9]
#LoadDkEast=LoadDkEast/np.mean(LoadDkEast)
WindDkEast=GW_reg[9]+GW_reg[10]
WindDkEast=WindDkEast/np.mean(WindDkEast)
SolarDkEast=GS_reg[9]
SolarDkEast=SolarDkEast/np.mean(SolarDkEast)
DenmarkEast=np.array([1000*LoadDkEast,WindDkEast,SolarDkEast])
np.save('DKE.npy',DenmarkEast)


LoadDkWest=L_reg[11]
#LoadDkWest=LoadDkWest/np.mean(LoadDkWest)
WindDkWest=GW_reg[11]+GW_reg[12]
WindDkWest=WindDkWest/np.mean(WindDkWest)
SolarDkWest=GS_reg[11]
SolarDkWest=SolarDkWest/np.mean(SolarDkWest)
DenmarkWest=np.array([1000*LoadDkWest,WindDkWest,SolarDkWest])
np.save('DKW.npy',DenmarkWest)

LoadNorge=L_reg[60]+L_reg[61]+L_reg[62]
WindNorge=GW_reg[60]+GW_reg[61]+GW_reg[62]+GW_reg[63]
WindNorge=WindNorge/np.mean(WindNorge)
SolarNorge=GS_reg[60]+GS_reg[61]+GS_reg[62]
SolarNorge=SolarNorge/np.mean(SolarNorge)
Norge=np.array([1000*LoadNorge,WindNorge,SolarNorge])
np.save('N.npy',Norge)

LoadSverige=L_reg[77]+L_reg[79]+L_reg[81]
WindSverige=GW_reg[77]+GW_reg[78]+GW_reg[79]+GW_reg[80]+GW_reg[81]+GW_reg[82]
WindSverige=WindSverige/np.mean(WindSverige)
SolarSverige=GS_reg[77]+GS_reg[79]+GS_reg[81]
SolarSverige=SolarSverige/np.mean(SolarSverige)
Sverige=np.array([1000*LoadSverige,WindSverige,SolarSverige])
np.save('S.npy',Sverige)


LoadDNord=L_reg[14]+L_reg[20]
WindDNord=GW_reg[14]+GW_reg[18]+GW_reg[17]+GW_reg[20]
WindDNord=WindDNord/np.mean(WindDNord)
SolarDNord=GS_reg[14]+GS_reg[20]
SolarDNord=SolarDNord/np.mean(SolarDNord)
DNord=np.array([1000*LoadDNord,WindDNord,SolarDNord])
np.save('DN.npy',DNord)



# Gets time series for all countries in normalized units.
#t, l, Gw, Gs, datetime_offset, datalabels = get_data_countries(schema='norm_agg_avg_1hour_pdata_caps_eu2020',localhost=True) 

#Sweden=np.array([l[23],Gw[23],Gs[23]])
#Germany=np.array([l[6],Gw[6],Gs[6]])
#Norway=np.array([l[18],Gw[18],Gs[18]])

#np.save('De.npy',Germany)
#np.save('No.npy',Norway)
#np.save('Se.npy',Sweden)
