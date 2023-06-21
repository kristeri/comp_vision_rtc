import numpy as np
import matplotlib.pyplot as plt

# For plotting the results

x = np.linspace(10, 60, 602)

y1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.008042335510253906, 0.008042335510253906, 0.008042335510253906, 0.008042335510253906, 0.008042335510253906, 0.008042335510253906, 0.010399031639099122, 0.010399031639099122, 0.010399031639099122, 0.010399031639099122, 0.010399031639099122, 0.010399031639099122, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.011108182668685913, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012568767249584199, 0.012033354315161705, 0.012033354315161705, 0.012033354315161705, 0.012033354315161705, 0.012033354315161705, 0.012033354315161705, 0.012033354315161705, 0.012033354315161705, 0.012033354315161705, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.010857847742289305, 0.013438629408918323, 0.013438629408918323, 0.013438629408918323, 0.013438629408918323, 0.013438629408918323, 0.01243220389406495, 0.01243220389406495, 0.01243220389406495, 0.01243220389406495, 0.01243220389406495, 0.01243220389406495, 0.01243220389406495, 0.01243220389406495, 0.01243220389406495, 
0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.011229127918109504, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.010475163393479016, 0.009732119265469125, 0.009732119265469125, 0.009732119265469125, 0.009732119265469125, 0.009732119265469125, 0.009732119265469125, 0.00922051590659358, 0.00922051590659358, 0.00922051590659358, 0.00922051590659358, 0.00922051590659358, 0.00922051590659358, 0.00922051590659358, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.009069502290898732, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.008514347619108405, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 0.012061452373214801, 
0.012061452373214801, 0.011367032109395667, 0.011367032109395667, 0.011367032109395667, 0.011367032109395667, 0.011367032109395667, 0.011367032109395667, 0.011367032109395667, 0.011367032109395667, 0.011367032109395667, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.010388033391924305, 0.009664603354143961, 0.009664603354143961, 0.009664603354143961, 0.009664603354143961, 0.009664603354143961, 0.015510521371530181, 0.015510521371530181, 0.015510521371530181, 0.015510521371530181, 0.015510521371530181, 0.015510521371530181, 0.015510521371530181, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.013993326558134638, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.022031825953168105, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.021065151539868182, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.018577826490314954, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.016445754567548963, 0.017484252462586294, 0.017484252462586294, 0.017484252462586294, 0.017484252462586294, 0.017484252462586294, 0.017484252462586294, 0.017484252462586294, 0.017484252462586294, 0.017484252462586294, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.019052977451016955, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.017721136237264557, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.015872170380074532, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.014719117058933714, 0.013691528755006985, 0.013691528755006985, 0.013691528755006985, 0.013691528755006985, 0.013691528755006985, 0.013691528755006985, 0.01481940001243465, 0.01481940001243465, 0.01481940001243465, 0.01481940001243465, 0.01481940001243465, 0.013639154061625362, 0.013639154061625362, 0.013639154061625362, 0.013639154061625362, 0.013639154061625362, 0.013639154061625362, 0.013639154061625362, 0.013639154061625362, 0.013639154061625362, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.015042530219959683, 0.013834536783920076, 0.013834536783920076, 0.013834536783920076, 0.013834536783920076, 0.013834536783920076, 0.013834536783920076, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.012650386101079379, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.015316124027134514, 0.013761963422484503, 0.013761963422484503, 0.013761963422484503, 0.013761963422484503, 0.013761963422484503, 0.013761963422484503, 0.013761963422484503, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.013924581884423106, 0.014787111299452512, 0.014787111299452512, 0.014787111299452512, 0.014787111299452512, 0.014787111299452512, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.013957284465069303, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012698538291890937, 0.012558615317119748, 0.012558615317119748, 0.012558615317119748, 0.012558615317119748, 0.012558615317119748, 0.012558615317119748, 0.012558615317119748, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.012202037069844754, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.015047644366423706, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 
0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.013763245514194526, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.012591326321647868, 0.01331749506062603, 0.01331749506062603, 0.01331749506062603, 0.01331749506062603, 0.01331749506062603, 0.01331749506062603, 0.01331749506062603, 0.013824839444720603, 0.013824839444720603, 0.013824839444720603, 0.013824839444720603, 0.013824839444720603, 0.013824839444720603, 0.013824839444720603, 0.013824839444720603, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012636349791287659, 0.012625131274163114, 0.012625131274163114, 0.012625131274163114, 0.012625131274163114, 0.012625131274163114, 0.012625131274163114, 0.012625131274163114, 0.012441394999786694, 0.012441394999786694, 0.012441394999786694, 0.012441394999786694, 0.012441394999786694, 0.013056193325807458, 0.013056193325807458, 0.013056193325807458, 0.013056193325807458, 0.013056193325807458, 
0.013056193325807458, 0.013056193325807458, 0.014466547324067687, 0.014466547324067687, 0.014466547324067687, 0.014466547324067687, 0.014466547324067687, 0.014466547324067687, 0.014466547324067687, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.013721533468102797, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.01413851745240398, 0.013804877819772876, 0.013804877819772876, 0.013804877819772876, 0.013804877819772876, 0.013804877819772876, 0.013804877819772876, 0.013804877819772876, 0.013804877819772876, 0.01442808535579132, 0.01442808535579132, 0.01442808535579132, 0.01442808535579132, 0.01442808535579132, 0.01442808535579132, 0.01829076044609938, 0.01829076044609938, 0.01829076044609938, 0.01829076044609938, 0.01829076044609938, 0.01636886793298696, 
0.01636886793298696, 0.01636886793298696, 0.01636886793298696, 0.01636886793298696, 0.01636886793298696, 0.01636886793298696, 0.01636886793298696, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.015094532253689553, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511, 0.013867580523088511]

y2 = [0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 
0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.016952037811279297, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015512442588806151, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.015061513185501097, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 
0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.01329677826166153, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.011466806104779244, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.013402171160131694, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.01239584996456653, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.011346213490084188, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.012768232424823512, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.011178438921573618, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 
0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010669971804651051, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010902604963091578, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.010543194691192538, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014870121268305895, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.014140352666072706, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 
0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.015093254348376155, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.014097521906721536, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.017008816872819018, 0.021782678687141527, 0.021782678687141527, 0.021782678687141527]

print(sum(y1) / len(y1))
print(len(y1))
print(len(y2))


y1 = [x * 1000 for x in y1]
y2 = [x * 1000 for x in y2]

plt.plot(x, y1, "royalblue", label="Latency Physical")
plt.plot(x, y2, "tomato", label="Latency Virtual")
plt.legend(loc="upper left")
#plt.ylim(-1.0, 2.0)
plt.xlabel("time (s)")
plt.ylabel("round-trip time (ms)")

plt.show()