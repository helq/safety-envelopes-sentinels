airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # m/s

means = []
stds = []
stall_prob = []

# airspeed = 6
means.append([30.415984247187414, 31.021954683393385, 25.931886822588297, 34.7191147012409, 26.61373471315814, 27.83467504223403, 23.649790679760244, 26.348454930469227, 26.33368652991115, 25.985009185300527, 26.578743643392272, 466.043906622781, 411.4504615398303, 355.08585882568406, 445.68841897824416, 559.5974875653089, 546.9847219685801, 417.0508742089745]) # noqa
stds.append([3.1884654617857353, 3.251989365383651, 2.7184060812312434, 3.639570879345529, 2.7898827133578274, 2.9178716684080332, 2.479175242418584, 2.7620720225427595, 2.7605263345499287, 2.7239756378841316, 2.786215758159038, 48.85523311667722, 43.132091242015534, 37.22332759863206, 46.72120157453888, 58.662159357613895, 57.34017399429725, 43.71964826733314]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
# airspeed = 8
means.append([36.713383689957205, 38.87995473433005, 50.56287128815728, 49.19247044216325, 41.31555731238518, 43.56873662994691, 40.57715212161447, 48.04616534046963, 45.377962251505494, 49.137223287747425, 47.89318489264653, 1118.5999168717424, 1210.7934123942666, 1296.260859180662, 1541.019502494365, 2335.566420467163, 1739.1138257363866, 985.2920851232981]) # noqa
stds.append([3.848641849868003, 4.075737026862258, 5.300669686499844, 5.156886397195031, 4.331061059689437, 4.5672687037898045, 4.2536649029221625, 5.036667783546087, 4.75691915510402, 5.15102085060086, 5.020592274710125, 117.27644339559384, 126.9267489241209, 135.88596903842762, 161.54465356300184, 244.83705885977776, 182.311469119549, 103.28744953196416]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
# airspeed = 10
means.append([59.54060797641883, 66.34898087113702, 77.2568390481836, 88.68767078619591, 75.3707225443223, 91.88907062935269, 101.91580497090527, 110.33483531774294, 109.18297986322092, 135.16787611806086, 129.70434624865393, 190.82984253991052, 557.5470336686793, 3978.9735124522017, 3679.4164337156776, 4747.200324119402, 11940.173073077933, 3884.6755004035185]) # noqa
stds.append([6.241571808713906, 6.955289422833688, 8.098745851507717, 9.29703850057509, 7.901021489248276, 9.632656588978092, 10.683755432178296, 11.566297963814781, 11.445538971650555, 14.16957324995162, 13.596790545777097, 20.004606301151473, 58.44708900068666, 417.1140536548297, 385.71166686503443, 497.64686805233157, 1251.6827478213525, 407.2343250323594]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# airspeed = 11
means.append([70.32801604109936, 87.05245570590695, 95.13135820095374, 115.67398949789653, 93.09492204298405, 137.65629948075892, 157.68244149545137, 156.53269060539452, 169.41496140285594, 188.6425770540385, 192.626482497043, 228.82033552364058, 917.5083813072355, 7824.138168732393, 6536.731630867065, 5892.093992278747, 25091.694068106826, 11677.537377720992]) # noqa
stds.append([7.37240370669988, 9.125655112913153, 9.972546448578798, 12.126007643102344, 9.759029705940716, 14.430393810786839, 16.529776930801994, 16.40914791778454, 17.759657798299507, 19.77520651788215, 20.192903878492036, 23.987024467487664, 96.1814037164303, 820.2120146962253, 685.2422432795348, 617.6665340432476, 2630.366859127947, 1224.1636589947746]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# airspeed = 12
means.append([113.94845444913011, 119.43494171227302, 143.77507926922948, 172.45171713471035, 152.99345409266502, 185.9376004786575, 226.62609409010776, 195.92905890212802, 233.06733231355076, 268.98072217488397, 308.3728048086663, 376.9044844023127, 1360.5607556627879, 13516.233788951662, 10973.31526938887, 8901.150224052411, 34130.06602970355, 23788.202885925893]) # noqa
stds.append([11.946019687482998, 12.520251933493366, 15.072263095196707, 18.077943621968714, 16.038119616069825, 19.491625436954997, 23.75711521783849, 20.539042195811405, 24.432230839713938, 28.196979937179666, 32.32640454294943, 39.510548198022875, 142.62625122253823, 1416.905126072939, 1150.3327677013847, 933.1091435433083, 3577.855019350262, 2493.737869241694]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# airspeed = 13
means.append([146.59419127245383, 180.90118502806556, 218.6119375456811, 269.2666414685412, 236.64819387245836, 273.9751254085315, 348.8102097151605, 325.42849990770924, 335.6962217860003, 412.74174813281917, 464.3320987664557, 618.8494339580849, 1876.5306935806539, 21197.522654071516, 18946.36779859625, 12460.935224890267, 36460.50037249014, 55641.55072293701]) # noqa
stds.append([15.367356439579194, 18.963668736661184, 22.916853805656263, 28.227011316785582, 24.807548046783815, 28.72051230355297, 36.56568252671931, 34.11442948058976, 35.190733213634225, 43.26745899427154, 48.67549960628646, 64.87331176542412, 196.71483436160307, 2222.1265885505345, 1986.1515626924584, 1306.2742886201363, 3822.1758453330085, 5832.919336889414]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# airspeed = 14
means.append([236.6811415728431, 262.16420990090865, 315.70417282881454, 369.42931753383584, 359.08815052379197, 419.01729835127117, 470.13870169666114, 469.68965166027135, 510.631420279458, 570.8074455983402, 665.1074755382139, 903.0957075762617, 2602.072847928006, 4601.009291456144, 26757.720381190415, 19984.280269396393, 35623.63451829266, 92821.47617855376]) # noqa
stds.append([24.811023127095066, 27.48237377705459, 33.09488693523339, 38.72690043811302, 37.6430125256239, 43.925141260690104, 49.2841963722785, 49.23714060640893, 53.52891745653005, 59.83771310029621, 69.72247440168962, 94.67055216154598, 272.7728991063241, 482.3200797120067, 2804.997458810925, 2094.942866321289, 3734.4432011165864, 9730.444664023897]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
# airspeed = 15
means.append([307.25204616317615, 425.58270013093477, 550.0519667564491, 616.0736332360711, 771.0953393522408, 734.0489874532848, 834.5729685778595, 851.9301332053059, 813.8155011340315, 858.246641607776, 1158.6029362807249, 1402.1589195775582, 3482.8595868732186, 6200.554105568745, 47538.13972271907, 30984.973889338315, 40416.35177449525, 108718.44068359418]) # noqa
stds.append([32.20894834821503, 44.61328330603728, 57.661296243382466, 64.58230118997749, 80.8329293615517, 76.94948155110889, 87.48724748224231, 89.30687985666488, 85.3113017709063, 89.9690391877446, 121.45497815139696, 146.9866867194344, 365.10461136629795, 649.9984273681746, 4983.385695042365, 3248.1360085409324, 4236.825387116742, 11396.92154196612]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
# airspeed = 16
means.append([376.6834023748227, 455.32372224940036, 500.98570831650466, 629.4445426259222, 561.8322556345748, 697.6781646418686, 742.3783481900416, 751.5887093861454, 812.2800494310625, 964.308863891398, 1342.9569261915992, 2108.3381232586016, 5326.6701481121445, 8543.357393026095, 8698.525859650596, 28322.08059590535, 47228.47738440653]) # noqa
stds.append([39.48735972424712, 47.731148857830185, 52.51770695982156, 65.9840716178154, 58.896186766059934, 73.13693257442097, 77.8228510004599, 78.7882649735719, 85.15056077409444, 101.08755416956357, 140.78066542017214, 221.01478646376697, 558.3895360748577, 895.5913271327024, 911.8571566563339, 2969.042891805068, 4950.934220815742]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
# airspeed = 17
means.append([455.51934899997025, 524.2078388288505, 618.1762806426236, 765.7836667224404, 724.232084542224, 847.9727925974768, 1070.0759745195692, 989.1946683648019, 1082.1136443649948, 1231.7687707426567, 1620.0553791288867, 2796.640867020935, 8025.890093910138, 11622.208239538757, 12299.578336571927, 12717.646085286317]) # noqa
stds.append([47.751590410432165, 54.95213075253732, 64.80273051589276, 80.27619908741937, 75.92044340703407, 88.89214002602175, 112.17507842484012, 103.69661306841532, 113.43693821351296, 129.1250967521603, 169.82861344527205, 293.169042448982, 841.3478104549818, 1218.3468989942985, 1289.354726232745, 1333.182749792903]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
# airspeed = 18
means.append([530.604162794817, 622.6640057959418, 763.3548200161091, 955.0891139000145, 933.276613667843, 1057.7349302142648, 1330.2330535634576, 1275.6207852023326, 1411.9691359536862, 1512.3684741946092, 2031.8052021209262, 3693.2417711916155, 9902.617641956174, 14952.301259183845]) # noqa
stds.append([55.622654970573876, 65.27313362080942, 80.0216167577235, 100.12099409374385, 97.83440365594073, 110.88131496952573, 139.44840018906862, 133.72207066625802, 148.0151832729986, 158.53997287995713, 212.9920628109035, 387.15930123700986, 1038.0817482978139, 1567.4384311090282]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
# airspeed = 19
means.append([625.1948929797995, 723.0013592289323, 989.990341210374, 1219.5558190316112, 1066.069516358782, 1326.3167931997857, 1499.6896527770084, 1640.2840801251311, 1784.4689152638882, 1953.4272789680936, 2491.857882520074, 4741.711128633746, 13384.41906727211]) # noqa
stds.append([65.53854982478342, 75.79137569267452, 103.7794876173269, 127.84472612556337, 111.7548288528611, 139.0364888681345, 157.21124779321508, 171.94942418806963, 187.06439776359872, 204.77571436086077, 261.218645572602, 497.06881211427276, 1403.0767836229513]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
# airspeed = 20
means.append([699.7137781028978, 876.9825025738336, 1089.2436628451796, 1371.5003148362691, 1332.3329351643954, 1542.3259677616184, 1824.7235860942487, 1779.4406525277952, 2035.4420352469501, 2352.678468360337, 2807.830818419733, 5800.276118223402]) # noqa
stds.append([73.35019807550636, 91.93307126264402, 114.18415713991908, 143.77265890035787, 139.66685784392251, 161.68031733907435, 191.2844395601159, 186.5368998780857, 213.37329988041807, 246.6292706329826, 294.34175959257044, 608.0375286341044]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
# airspeed = 21
means.append([861.5746461724494, 1025.3633625027135, 1233.4426254640618, 1498.4372354297495, 1515.852349481205, 1802.0709319948262, 1978.8827656473304, 2254.5187586277593, 2328.6151100497127]) # noqa
stds.append([90.31787179621702, 107.4876086238094, 129.30031804183065, 157.07936991057304, 158.90506796634756, 188.90923431341463, 207.44432746981482, 236.33974704551954, 244.10626266802603]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
# airspeed = 22
means.append([1015.542453005997, 1210.9180138053855, 1521.5412259964057, 1838.6056752138486, 1699.751405684395, 1844.4813336740012, 2304.8809827997334]) # noqa
stds.append([106.45819801840108, 126.93918625963087, 159.50150647753543, 192.73897607283575, 178.18310633241552, 193.35490769796812, 241.61822954743116]) # noqa
stall_prob.append([0, 0, 0, 0, 0, 0, 0])