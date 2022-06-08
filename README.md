# Solving LunarLander-v2

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

This repository contains my solution for the [LunarLander-v2](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) OpenAI gym environment.
The code is written in Python using libraries from the 
[DeepMind JAX ecosystem](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research).

The code solves the LunarLander-v2 environment in approximately 40&thinsp;000&mdash;50&thinsp;000 frames
(about 120 episodes).
I count the environment as solved once the average reward
(measured over the preceeding 100 episodes)
exceeds 200.

## References

* Chen, Xinyue, Che Wang, Zijian Zhou, and Keith Ross. 2021. "Randomized
Ensembled Double Q-Learning: Learning Fast Without a Model." arXiv.
<https://doi.org/10.48550/ARXIV.2101.05982>.

* Ferret, Johan, Olivier Pietquin, and Matthieu Geist. 2020.
"Self-Imitation Advantage Learning." arXiv.
<https://doi.org/10.48550/ARXIV.2012.11989>.

* Lee, Kimin, Michael Laskin, Aravind Srinivas, and Pieter Abbeel. 2020.
"SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep
Reinforcement Learning." arXiv.
<https://doi.org/10.48550/ARXIV.2007.04938>.

* Sinha, Samarth, Homanga Bharadhwaj, Aravind Srinivas, and Animesh Garg. 2020. 
"D2rl: Deep Dense Architectures in Reinforcement Learning." arXiv.
<https://doi.org/10.48550/ARXIV.2010.09163>.

## Log

<details>
<summary>Program output (default parameters)</summary>

```
E: 1, Frames: 80, R: -459.16193741701045, T: -459.16193741701045, L: 0.0, Total frames: 80
E: 2, Frames: 74, R: -569.099718720431, T: -514.1308280687208, L: 0.0, Total frames: 154
E: 3, Frames: 70, R: -482.66849751478816, T: -503.6433845507433, L: 0.0, Total frames: 224
E: 4, Frames: 76, R: -503.1463754108974, T: -503.5191322657818, L: 0.0, Total frames: 300
E: 5, Frames: 85, R: -522.9989408322349, T: -507.4150939790725, L: 0.0, Total frames: 385
E: 6, Frames: 121, R: -332.268572384515, T: -478.22400704664625, L: 0.0, Total frames: 506
E: 7, Frames: 76, R: -380.055974160834, T: -464.2000023486731, L: 0.0, Total frames: 582
E: 8, Frames: 57, R: -136.35458832620705, T: -423.21932559586486, L: 0.0, Total frames: 639
E: 9, Frames: 89, R: -422.8251913002291, T: -423.17553289634975, L: 0.0, Total frames: 728
E: 10, Frames: 91, R: -79.2017406270572, T: -388.7781536694205, L: 1.94221509007944, Total frames: 819
E: 11, Frames: 181, R: -175.19821252880058, T: -369.36179538390957, L: 1.8218549581313574, Total frames: 1000
E: 12, Frames: 91, R: -197.10115647721653, T: -355.00674214168515, L: 1.7968226687568758, Total frames: 1091
E: 13, Frames: 101, R: -283.6309820203809, T: -349.5162990554309, L: 1.7071951307518327, Total frames: 1192
E: 14, Frames: 82, R: -446.50464928843905, T: -356.44403835778866, L: 1.6947621687585677, Total frames: 1274
E: 15, Frames: 795, R: -226.3674424158816, T: -347.7722652949949, L: 1.786839302018285, Total frames: 2069
E: 16, Frames: 304, R: -40.609409725466975, T: -328.5745868218994, L: 2.355591029345989, Total frames: 2373
E: 17, Frames: 1000, R: 5.112203842091175, T: -308.9459520769588, L: 1.852382513642311, Total frames: 3373
E: 18, Frames: 911, R: 134.84597746211082, T: -284.29084488034385, L: 1.5667053992450237, Total frames: 4284
E: 19, Frames: 1000, R: -9.997887058661393, T: -269.8543734160447, L: 2.1241984628140926, Total frames: 5284
E: 20, Frames: 564, R: -74.14483931764968, T: -260.068896711125, L: 1.6220996130108833, Total frames: 5848
E: 21, Frames: 209, R: 15.27438817214221, T: -246.95731171668368, L: 1.5649262246787548, Total frames: 6057
E: 22, Frames: 773, R: -163.34549256100956, T: -243.15677448233484, L: 1.4733092225193978, Total frames: 6830
E: 23, Frames: 1000, R: -68.84359670065932, T: -235.57794066574024, L: 1.3658310364335775, Total frames: 7830
E: 24, Frames: 525, R: 233.0029983776116, T: -216.05373487226726, L: 1.2979012784212827, Total frames: 8355
E: 25, Frames: 326, R: -90.5480248982525, T: -211.03350647330666, L: 1.5406372046917678, Total frames: 8681
E: 26, Frames: 249, R: -119.44875131577626, T: -207.51101589032473, L: 1.665306094750762, Total frames: 8930
E: 27, Frames: 1000, R: 11.44018037783208, T: -199.40171232483746, L: 1.4378294439017774, Total frames: 9930
E: 28, Frames: 503, R: 240.2762074506302, T: -183.6989294757136, L: 1.4119521288126706, Total frames: 10433
E: 29, Frames: 256, R: 43.03004275356446, T: -175.88068905401434, L: 1.5818243072926998, Total frames: 10689
E: 30, Frames: 443, R: 193.0095289613749, T: -163.58434845350138, L: 1.7557728471457958, Total frames: 11132
E: 31, Frames: 663, R: 169.8128621987662, T: -152.8295997227831, L: 1.6788542597591878, Total frames: 11795
E: 32, Frames: 694, R: 182.82425015213164, T: -142.340416914192, L: 1.5433673797547818, Total frames: 12489
E: 33, Frames: 145, R: -61.707415093201064, T: -139.8969926165862, L: 1.5438627014756203, Total frames: 12634
E: 34, Frames: 435, R: 240.9939624818989, T: -128.69431746663076, L: 1.528308257251978, Total frames: 13069
E: 35, Frames: 453, R: 263.43279322813555, T: -117.49068573249458, L: 1.6301862812638284, Total frames: 13522
E: 36, Frames: 387, R: 245.14591233249442, T: -107.417446897356, L: 1.6773167144060135, Total frames: 13909
E: 37, Frames: 485, R: -125.63164888524426, T: -107.90972262675838, L: 1.702325455069542, Total frames: 14394
E: 38, Frames: 1000, R: -11.650575506728096, T: -105.37658717623128, L: 1.5445521330237388, Total frames: 15394
E: 39, Frames: 201, R: -53.26898660340734, T: -104.04049485385119, L: 1.5063882530629635, Total frames: 15595
E: 40, Frames: 340, R: 231.6385366961106, T: -95.64851906510214, L: 1.4802348716557026, Total frames: 15935
E: 41, Frames: 326, R: 216.8710704272355, T: -88.0260900530939, L: 1.485629316687584, Total frames: 16261
E: 42, Frames: 635, R: 172.97243594444137, T: -81.81183943410495, L: 1.5561943553686142, Total frames: 16896
E: 43, Frames: 281, R: 243.4161782363669, T: -74.24839716269864, L: 1.6121578358113766, Total frames: 17177
E: 44, Frames: 413, R: 266.3046403289559, T: -66.50855540152467, L: 1.727691042214632, Total frames: 17590
E: 45, Frames: 377, R: 215.2662268460509, T: -60.246893573800776, L: 1.8273623929023743, Total frames: 17967
E: 46, Frames: 262, R: 262.1742565367875, T: -53.23773813661407, L: 1.9009150269031525, Total frames: 18229
E: 47, Frames: 258, R: 238.1131900655856, T: -47.03878221741833, L: 1.9578896215558053, Total frames: 18487
E: 48, Frames: 257, R: 203.20805000171737, T: -41.82530654618634, L: 2.0474600082933905, Total frames: 18744
E: 49, Frames: 265, R: 242.18582529324559, T: -36.029160998442826, L: 2.1069359062314033, Total frames: 19009
E: 50, Frames: 253, R: 254.42794925655812, T: -30.22001879334281, L: 2.1469213883280753, Total frames: 19262
E: 51, Frames: 283, R: 281.0419592148926, T: -24.116842753965646, L: 2.112478666096926, Total frames: 19545
E: 52, Frames: 191, R: 50.10850099266429, T: -22.68943229729969, L: 2.050313155412674, Total frames: 19736
E: 53, Frames: 172, R: -70.30698535293476, T: -23.587876694575822, L: 2.002930963039398, Total frames: 19908
E: 54, Frames: 261, R: 287.8059624857385, T: -17.821324117162593, L: 1.9183843481838703, Total frames: 20169
E: 55, Frames: 247, R: 298.77963332702797, T: -12.064943072722764, L: 1.8777435404658318, Total frames: 20416
E: 56, Frames: 258, R: 263.1360129034885, T: -7.150640287433277, L: 1.8514787299036979, Total frames: 20674
E: 57, Frames: 628, R: 256.5824574981823, T: -2.523743835054057, L: 1.83427114123106, Total frames: 21302
E: 58, Frames: 263, R: 273.29986375176816, T: 2.23183560609805, L: 1.8262690472900867, Total frames: 21565
E: 59, Frames: 256, R: 283.14268509327746, T: 6.993036444863803, L: 1.7989215907752514, Total frames: 21821
E: 60, Frames: 257, R: 218.31378396328228, T: 10.515048903504113, L: 1.7493864262998104, Total frames: 22078
E: 61, Frames: 927, R: 184.17576336381993, T: 13.361945861869945, L: 1.6646515168845653, Total frames: 23005
E: 62, Frames: 238, R: 260.9267656980295, T: 17.354926826969294, L: 1.6437730807363986, Total frames: 23243
E: 63, Frames: 193, R: 52.95085297724012, T: 17.919941527767243, L: 1.6284804020226002, Total frames: 23436
E: 64, Frames: 251, R: 220.48028661672603, T: 21.084946919782226, L: 1.6044481942653657, Total frames: 23687
E: 65, Frames: 874, R: 242.32500199486262, T: 24.48864007478346, L: 1.6264910431206225, Total frames: 24561
E: 66, Frames: 262, R: 289.9123909487653, T: 28.51021205772258, L: 1.6479994603097439, Total frames: 24823
E: 67, Frames: 570, R: 238.28371634637094, T: 31.641159882926285, L: 1.6518406894803048, Total frames: 25393
E: 68, Frames: 269, R: 286.36591007534105, T: 35.387112091638265, L: 1.6662835786938668, Total frames: 25662
E: 69, Frames: 570, R: 254.63961548292468, T: 38.56468460455547, L: 1.744124408364296, Total frames: 26232
E: 70, Frames: 236, R: 301.2357711132049, T: 42.31712869753617, L: 1.7652054781019688, Total frames: 26468
E: 71, Frames: 246, R: 261.12019159640823, T: 45.39886197780198, L: 1.795348512649536, Total frames: 26714
E: 72, Frames: 212, R: 20.82668456085628, T: 45.05758173589996, L: 1.8072033067345619, Total frames: 26926
E: 73, Frames: 248, R: 249.18779240676403, T: 47.853885991665216, L: 1.8430516689717769, Total frames: 27174
E: 74, Frames: 318, R: 272.81098044821516, T: 50.89384672756454, L: 1.8880474636256694, Total frames: 27492
E: 75, Frames: 283, R: 302.7055367182509, T: 54.25133592744036, L: 1.8955592812895774, Total frames: 27775
E: 76, Frames: 239, R: 244.89143253518017, T: 56.75975825122641, L: 1.883800339192152, Total frames: 28014
E: 77, Frames: 260, R: 278.01783408842243, T: 59.63323975560558, L: 1.8722491292357444, Total frames: 28274
E: 78, Frames: 291, R: 218.6858310935399, T: 61.67237554198935, L: 1.8535682265460491, Total frames: 28565
E: 79, Frames: 249, R: 291.0579419241036, T: 64.57599030631991, L: 1.8528253372013568, Total frames: 28814
E: 80, Frames: 342, R: 245.55088374460868, T: 66.83817647429852, L: 1.8575574583113192, Total frames: 29156
E: 81, Frames: 293, R: 276.27199921306305, T: 69.42377922415982, L: 1.8417115260362624, Total frames: 29449
E: 82, Frames: 149, R: -13.943651802658792, T: 68.40710323602788, L: 1.8477929584383965, Total frames: 29598
E: 83, Frames: 244, R: 227.11492529637292, T: 70.31924567048988, L: 1.8421818379461765, Total frames: 29842
E: 84, Frames: 224, R: 253.5461426858612, T: 72.5005182540062, L: 1.8461812032163143, Total frames: 30066
E: 85, Frames: 213, R: 78.088253803306, T: 72.56625631929207, L: 1.8649813881814479, Total frames: 30279
E: 86, Frames: 218, R: 250.34100026939282, T: 74.63340450475836, L: 1.8758255685567855, Total frames: 30497
E: 87, Frames: 453, R: 8.23668829121415, T: 73.87022385862566, L: 1.8699038005173205, Total frames: 30950
E: 88, Frames: 1000, R: 117.0916905229881, T: 74.3613768889025, L: 1.875452271282673, Total frames: 31950
E: 89, Frames: 229, R: 290.576400306501, T: 76.7907591744935, L: 1.8365982067286968, Total frames: 32179
E: 90, Frames: 238, R: 242.00954945787984, T: 78.62652351097557, L: 1.8198625945746898, Total frames: 32417
E: 91, Frames: 271, R: 286.1692235084711, T: 80.90721252193707, L: 1.828795248299837, Total frames: 32688
E: 92, Frames: 217, R: 232.2808027716962, T: 82.5525776333475, L: 1.839150535017252, Total frames: 32905
E: 93, Frames: 345, R: 275.92742091738586, T: 84.63187702349843, L: 1.880592763900757, Total frames: 33250
E: 94, Frames: 239, R: 236.4081905077041, T: 86.24651865630914, L: 1.8832554847598075, Total frames: 33489
E: 95, Frames: 411, R: -26.77564567097913, T: 85.0568116633903, L: 1.8643233618438244, Total frames: 33900
E: 96, Frames: 236, R: 280.97675744493665, T: 87.09764443194808, L: 1.8791575511693954, Total frames: 34136
E: 97, Frames: 244, R: 257.5266789199828, T: 88.85464478749483, L: 1.865852343171835, Total frames: 34380
E: 98, Frames: 242, R: 256.21529772412873, T: 90.56240655215437, L: 1.870705846965313, Total frames: 34622
E: 99, Frames: 224, R: 262.29040259724513, T: 92.29703277483206, L: 1.8791689555346967, Total frames: 34846
E: 100, Frames: 237, R: 226.59088012685532, T: 93.63997124835231, L: 1.8524062452018262, Total frames: 35083
E: 101, Frames: 211, R: 272.39207596657945, T: 100.95551138218822, L: 1.8347224700152873, Total frames: 35294
E: 102, Frames: 255, R: 253.08945973301996, T: 109.17740316672273, L: 1.8411964844465256, Total frames: 35549
E: 103, Frames: 243, R: 266.8325427371799, T: 116.6724135692424, L: 1.8387987958192826, Total frames: 35792
E: 104, Frames: 261, R: 276.98193970029354, T: 124.47369672035431, L: 1.866067524731159, Total frames: 36053
E: 105, Frames: 259, R: 265.3240353700312, T: 132.35692648237696, L: 1.8665719558000564, Total frames: 36312
E: 106, Frames: 260, R: 13.615583537220402, T: 135.81576804159434, L: 1.8696877424418927, Total frames: 36572
E: 107, Frames: 277, R: 293.8153543325259, T: 142.55448132652793, L: 1.876223756879568, Total frames: 36849
E: 108, Frames: 213, R: 271.41189705603426, T: 146.63214618035033, L: 1.877234268963337, Total frames: 37062
E: 109, Frames: 223, R: 266.35898807533636, T: 153.523987974106, L: 1.874455613642931, Total frames: 37285
E: 110, Frames: 259, R: 291.7441495439999, T: 157.23344687581658, L: 1.8788016096651554, Total frames: 37544
E: 111, Frames: 238, R: 285.46106270375594, T: 161.84003962814214, L: 1.8827231170535088, Total frames: 37782
E: 112, Frames: 263, R: 295.9713644953124, T: 166.77076483786743, L: 1.8615731583833695, Total frames: 38045
E: 113, Frames: 260, R: 273.38785530408626, T: 172.3409532111121, L: 1.8530179993510247, Total frames: 38305
E: 114, Frames: 216, R: 249.82097452574683, T: 179.304209449254, L: 1.8560516590476035, Total frames: 38521
E: 115, Frames: 210, R: 261.1036595549991, T: 184.1789204689628, L: 1.844343495041132, Total frames: 38731
E: 116, Frames: 280, R: 304.45349401080114, T: 187.6295495063255, L: 1.8416287028491498, Total frames: 39011
E: 117, Frames: 285, R: 316.050055500979, T: 190.73892802291437, L: 1.854746503174305, Total frames: 39296
E: 118, Frames: 217, R: 251.08517649776735, T: 191.90132001327092, L: 1.8446486208736896, Total frames: 39513
E: 119, Frames: 265, R: 277.6028672584768, T: 194.7773275564423, L: 1.8368225710093975, Total frames: 39778
E: 120, Frames: 1000, R: 129.42195881177858, T: 196.81299553773658, L: 1.781815176129341, Total frames: 40778
E: 121, Frames: 334, R: 242.9353208168778, T: 199.08960486418394, L: 1.751995687752962, Total frames: 41112
E: 122, Frames: 219, R: 270.31773543404813, T: 203.4262371441345, L: 1.7288535299897194, Total frames: 41331
```

</details>
