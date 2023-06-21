import numpy as np
import matplotlib.pyplot as plt

# For plotting the results

x = np.linspace(0, 60, 606)

y1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.004144191741943359, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.007348573207855225, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.006974167227745056, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.01155649682879448, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.017716620384156705, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.016005553718134762, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01748097760316357, 0.01577138999512922, 0.01577138999512922, 0.01577138999512922, 0.01577138999512922, 0.01577138999512922, 0.01577138999512922, 0.01577138999512922, 0.01577138999512922, 0.0191552289773967, 0.0191552289773967, 0.0191552289773967, 0.0191552289773967, 0.0191552289773967, 0.0191552289773967, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.017864841340229334, 0.015870008270451035, 0.015870008270451035, 0.015870008270451035, 0.015870008270451035, 0.015870008270451035, 0.015870008270451035, 0.015870008270451035, 0.015870008270451035, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.014165387938849442, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.015999806356298392, 0.01828172064531579, 0.01828172064531579, 0.01828172064531579, 0.01828172064531579, 0.01828172064531579, 0.017022581082606555, 0.017022581082606555, 0.017022581082606555, 0.017022581082606555, 0.017022581082606555, 0.017022581082606555, 0.017022581082606555, 0.017022581082606555, 0.017022581082606555, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.015515827640613522, 0.014633448897811287, 0.014633448897811287, 0.014633448897811287, 0.014633448897811287, 0.014633448897811287, 0.014633448897811287, 0.014633448897811287, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.013553336443663275, 0.012833044831789077, 0.012833044831789077, 0.012833044831789077, 0.012833044831789077, 0.012833044831789077, 0.012833044831789077, 0.012833044831789077, 0.012833044831789077, 0.017372462172419886, 0.017372462172419886, 0.017372462172419886, 0.017372462172419886, 0.017372462172419886, 0.017372462172419886, 0.017372462172419886, 0.017372462172419886, 0.018924124108000998, 0.018924124108000998, 0.018924124108000998, 0.018924124108000998, 0.018924124108000998, 0.018924124108000998, 0.018924124108000998, 0.018924124108000998, 0.022481000429697577, 0.022481000429697577, 0.022481000429697577, 0.022481000429697577, 
0.022481000429697577, 0.022481000429697577, 0.022481000429697577, 0.022481000429697577, 0.019872314339241963, 0.019872314339241963, 0.019872314339241963, 0.019872314339241963, 0.019872314339241963, 0.019872314339241963, 0.019872314339241963, 0.019872314339241963, 0.018985628698823198, 0.018985628698823198, 0.018985628698823198, 0.018985628698823198, 0.018985628698823198, 0.018985628698823198, 0.018985628698823198, 0.018985628698823198, 0.018985628698823198, 0.018578201206896447, 0.018578201206896447, 0.018578201206896447, 0.018578201206896447, 0.018578201206896447, 0.018578201206896447, 0.01683767559281755, 0.01683767559281755, 0.01683767559281755, 0.01683767559281755, 0.01683767559281755, 0.01683767559281755, 0.01683767559281755, 0.01683767559281755, 0.01683767559281755, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.018006785053302873, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.016726908919758126, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.018980331383063938, 0.021418806997564797, 0.021418806997564797, 0.021418806997564797, 0.021418806997564797, 0.021418806997564797, 0.021418806997564797, 0.023189530297600486, 0.023189530297600486, 0.023189530297600486, 0.023189530297600486, 0.023189530297600486, 0.023189530297600486, 0.023189530297600486, 0.023189530297600486, 0.020616292651306357, 0.020616292651306357, 0.020616292651306357, 0.020616292651306357, 0.020616292651306357, 0.020616292651306357, 0.020616292651306357, 0.020616292651306357, 0.020616292651306357, 0.0181998011881502, 0.0181998011881502, 0.0181998011881502, 0.0181998011881502, 0.0181998011881502, 0.0181998011881502, 0.0181998011881502, 0.01893352844339935, 0.01893352844339935, 0.01893352844339935, 0.01893352844339935, 0.01893352844339935, 0.018029052727609664, 
0.018029052727609664, 0.018029052727609664, 0.018029052727609664, 0.018029052727609664, 0.018029052727609664, 0.018029052727609664, 0.018029052727609664, 0.018029052727609664, 0.018029052727609664, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.01630928010365498, 0.0151229541205164, 0.0151229541205164, 0.0151229541205164, 0.0151229541205164, 0.0151229541205164, 0.0151229541205164, 0.0151229541205164, 0.0151229541205164, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.014871210316556372, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 
0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.01430264005142155, 0.013048703031898016, 0.013048703031898016, 0.013048703031898016, 0.013048703031898016, 0.012050197892970247, 0.012050197892970247, 0.012050197892970247, 0.012050197892970247, 0.012050197892970247, 0.012050197892970247, 0.012050197892970247, 0.012050197892970247, 0.01158194881434087, 0.01158194881434087, 0.01158194881434087, 0.01158194881434087, 0.01158194881434087, 0.01158194881434087, 0.01158194881434087, 0.01158194881434087, 0.01158194881434087, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.011892791295961713, 0.010793336579381176, 0.010793336579381176, 0.010793336579381176, 0.010793336579381176, 0.010793336579381176, 0.010793336579381176, 0.010794569151403442, 0.010794569151403442, 0.010794569151403442, 0.010794569151403442, 0.010794569151403442, 0.010794569151403442, 0.010794569151403442, 0.010794569151403442, 0.010794569151403442, 0.01110553714859649, 0.01110553714859649, 0.01110553714859649, 0.01110553714859649, 0.01110553714859649, 0.01110553714859649, 0.01110553714859649, 0.01110553714859649, 0.010501861346204478, 0.010501861346204478, 0.010501861346204478, 0.010501861346204478, 0.010501861346204478, 0.010501861346204478, 0.010501861346204478, 0.010501861346204478, 0.010501861346204478, 0.010943674849046755, 0.010943674849046755, 0.010943674849046755, 0.010943674849046755, 0.010943674849046755, 0.010943674849046755, 0.010943674849046755, 0.010943674849046755, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.016011115149247114, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.015251675049711609, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.013663730005633774, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.012979915572385143, 0.011878789671463896, 0.011878789671463896, 0.011878789671463896, 0.011878789671463896, 0.011878789671463896, 0.011878789671463896, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.014415113158000172, 0.017206850472019874, 0.017206850472019874, 0.017206850472019874, 0.017206850472019874, 0.017206850472019874, 0.017206850472019874, 0.017206850472019874, 0.015575003027407079, 0.015575003027407079, 0.015575003027407079, 0.015575003027407079, 0.015575003027407079, 0.015575003027407079, 0.015575003027407079, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.015893924921318477, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 
0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.014427545056869485, 0.018155117857665236, 0.018155117857665236, 0.018155117857665236, 0.018155117857665236, 0.018155117857665236, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.01639998458224421, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.015155206392588242, 0.019787397712294978, 0.019787397712294978, 0.019787397712294978, 0.019787397712294978, 0.019787397712294978, 0.019787397712294978, 0.019787397712294978, 0.019787397712294978, 0.02137804202273589, 0.02137804202273589, 0.02137804202273589]

y2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.004913330078125, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.006836831569671631, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.007234057784080506, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006491985768079757, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 
0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.006160559080541133, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 
0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.0058627889048308126, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 
0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006309347417708486, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006537967430296354, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.006177470565568795, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.005639627236745439, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006756201843250225, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 
0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.006448478639974361, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.00574256129038934, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.0056244708590379704, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.00508746612755166, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.0047139102457419094, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.00804898421989747, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067, 0.010749686647185067]
print(len(y1))
print(len(y2))

y1 = [x * 1000 for x in y1]
y2 = [x * 1000 for x in y2]

plt.plot(x, y1, "royalblue", label="Latency Physical")
plt.plot(x, y2, "tomato", label="Latency Virtual")
plt.legend(loc="upper right")
#plt.ylim(-1.0, 2.0)
plt.xlabel("time (s)")
plt.ylabel("round-trip time (ms)")

plt.show()
