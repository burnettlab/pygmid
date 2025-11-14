import logging
import os
import subprocess
from functools import cached_property
from pathlib import Path
from time import sleep
from typing import Tuple

import numpy as np
import pandas as pd

from .sim import Simulator, multiline_join


class NGSpiceSimulator(Simulator):
    """ NGSPICE simulator class for technology sweeps. """
    netlist_ext: str = 'spice'

    def __post_init__(self):
        self.args = [
            "-b",
            "-a",
            "-o",
            self.logfile,
            self.netlist_filepath,
        ]

    @property
    def output(self) -> str:
        return os.path.expandvars(f"techsweep_{'_'.join(self._config['MODEL']['MODELN'].split('_')[:-1])}.txt")
    
    @output.setter
    def output_setter(self, args: Tuple):
        pass

    @property
    def logfile(self) -> str:
        return self.output.replace('.txt', '.log')

    @property
    def symbol_dir(self) -> str:
        return self._config['SYMBOLS']['PATH']
    
    @cached_property
    def schematic_filepath(self) -> str:
        return self.netlist_filepath.replace(self.netlist_ext, "sch")

    def generate_netlist(self, **kwargs) -> str:
        sweep_codes = [
        f"""
        .param wx={kwargs['width']/kwargs['NFING']:.3f}u lx={kwargs['LEN_VEC'][0]:.3f}u
        .noise v(nn) vgn lin 1 1 1 1
        .noise v(np) vgp lin 1 1 1 1
        .op

        .control
        set wr_singlescale
        set wr_vecnames

        compose l_vec values {'u '.join(map(lambda l: f'{l:.3f}', kwargs['LEN_VEC']))}u
        compose vg_vec start= 0 stop={kwargs['VGS_max']+0.001:.3f}  step={kwargs['VGS_step']:.3f}
        compose vd_vec start= 0 stop={kwargs['VDS_max']+0.001:.3f}  step={kwargs['VDS_step']:.3f}
        compose vb_vec start= 0 stop=-{kwargs['VSB_max']+0.001:.3f} step=-{kwargs['VSB_step']:.3f}

        foreach var1 $&l_vec
            alterparam lx=$var1
            reset
            foreach var2 $&vg_vec
                alter vgn $var2
                alter vgp $var2
                foreach var3 $&vd_vec
                    alter vdn $var3
                    alter vdp $var3
                    foreach var4 $&vb_vec
                        alter vbn $var4
                        alter vbp $var4
                        run
                        wrdata {self.output} all
                        destroy all
                        set appendwrite
                        unset set wr_vecnames  
                    end
                end 
            end
        end
        unset appendwrite

        alterparam lx={kwargs['LEN_VEC'][0]:.3f}u
        reset
        op
        show
        write pysweep.raw
        .endc""",
        "\n".join(map(lambda l: "        .save @" + f"{kwargs['modeln']}[".join(l[0].split(':')) + "]", self._config['n'] + self._config['n_noise']))
        # + "\n".join(map(lambda l: "        .save onoise." + f"{kwargs['modeln']}[".join(l[0].split(':')) + "]", self._config['n_noise']))
        + f"""
        .save @vbn[dc]
        .save @vdn[dc]
        .save @vgn[dc]
        .save gn dn bn nn\n""",
        "\n".join(map(lambda l: "        .save @" + f"{kwargs['modelp']}[".join(l[0].split(':')) + "]", self._config['p'] + self._config['p_noise']))
        # + "\n".join(map(lambda l: "        .save onoise." + f"{kwargs['modeln']}[".join(l[0].split(':')) + "]", self._config['p_noise']))
        + f"""
        .save @vbp[dc]
        .save @vdp[dc]
        .save @vgp[dc]
        .save gp dp bp np\n""",
        """
        .control
        quit
        .endc
        """
        ]

        xschem_rc = self._config['SIMULATOR'].get('XSCHEM_RC', None)
        if xschem_rc is not None:
            comp_codes = [
            """
            "}
            C {devices/launcher.sym} 1000 -480 0 0 {name=h3
            descr="save, netlist & simulate"
            tclcommand="xschem save; xschem netlist; xschem simulate"}
            C {devices/code_shown.sym} 0 -940 0 0 {name=COMMANDS2 only_toplevel=false
            value="
            """,
            """
            "}
            C {devices/code_shown.sym} 640 -940 0 0 {name=COMMANDS3 only_toplevel=false
            value="
            """,
            "\"}"
            ]

            schem = """v {xschem version=3.4.6 file_version=1.2}
            G {}
            K {}
            V {}
            S {}
            E {}
            N 620 -390 620 -370 {lab=dn}
            N 620 -390 810 -390 {lab=dn}
            N 810 -390 810 -280 {lab=dn}
            N 500 -340 500 -280 {lab=gn}
            N 500 -340 580 -340 {lab=gn}
            N 720 -340 720 -280 {lab=bn}
            N 620 -340 720 -340 {lab=bn}
            N 890 -320 890 -280 {lab=nn}
            N 500 -220 500 -190 {lab=0}
            N 810 -220 810 -190 {lab=0}
            N 720 -220 720 -190 {lab=0}
            N 620 -310 620 -190 {lab=0}
            N 890 -220 890 -190 {lab=0}
            N 810 -190 890 -190 {lab=0}
            N 720 -190 810 -190 {lab=0}
            N 620 -190 720 -190 {lab=0}
            N 500 -190 620 -190 {lab=0}

            N 1620 -220 1620 -200 {lab=dp}
            N 1620 -200 1810 -200 {lab=dp}
            N 1810 -200 1810 -310 {lab=dp}
            N 1500 -310 1500 -250 {lab=gp}
            N 1500 -250 1580 -250 {lab=gp}
            N 1620 -250 1720 -250 {lab=bp}
            N 1720 -250 1720 -310 {lab=bp}
            N 1890 -310 1890 -270 {lab=np}
            N 1500 -370 1500 -390 {lab=0}
            N 1620 -280 1620 -390 {lab=0}
            N 1720 -370 1720 -390 {lab=0}
            N 1810 -370 1810 -390 {lab=0}
            N 1890 -370 1890 -390 {lab=0}
            N 1500 -390 1620 -390 {lab=0}
            N 1620 -390 1720 -390 {lab=0}
            N 1720 -390 1810 -390 {lab=0}
            N 1810 -390 1890 -390 {lab=0}

            C {devices/vsource.sym} 500 -250 0 0 {name=vgn value="DC 0.6 AC 1" savecurrent=false}
            C {devices/vsource.sym} 810 -250 0 0 {name=vdn value=0.6 savecurrent=false}
            C {devices/vsource.sym} 720 -250 0 1 {name=vbn value=0 savecurrent=false}
            C {devices/vsource.sym} 1500 -340 0 0 {name=vgp value="DC 0.6 AC 1" savecurrent=false}
            C {devices/vsource.sym} 1810 -340 0 0 {name=vdp value=0.6 savecurrent=false}
            C {devices/vsource.sym} 1720 -340 0 1 {name=vbp value=0 savecurrent=false}
            C {devices/lab_wire.sym} 550 -340 0 0 {name=p1 sig_type=std_logic lab=gn}
            C {devices/lab_wire.sym} 720 -390 0 0 {name=p2 sig_type=std_logic lab=dn}
            C {devices/lab_wire.sym} 720 -340 0 0 {name=p3 sig_type=std_logic lab=bn}
            C {devices/lab_wire.sym} 890 -320 0 0 {name=p4 sig_type=std_logic lab=nn}
            C {devices/lab_wire.sym} 550 -190 0 0 {name=p5 sig_type=std_logic lab=0}
            C {devices/lab_wire.sym} 1550 -250 0 0 {name=p6 sig_type=std_logic lab=gp}
            C {devices/lab_wire.sym} 1720 -200 0 0 {name=p7 sig_type=std_logic lab=dp}
            C {devices/lab_wire.sym} 1720 -250 0 0 {name=p8 sig_type=std_logic lab=bp}
            C {devices/lab_wire.sym} 1890 -280 0 0 {name=p9 sig_type=std_logic lab=np}
            C {devices/lab_wire.sym} 1550 -390 0 0 {name=p10 sig_type=std_logic lab=0}
            C {devices/code_shown.sym} 0 -1800 0 0 {name=COMMANDS1 only_toplevel=false
            value="

            """ + '\n'.join(''.join(s) for s in zip(sweep_codes, comp_codes)) + """
            C {devices/ccvs.sym} 890 -250 0 0 {name=Hn vnam=vdn value=1}
            C {devices/ccvs.sym} 1890 -340 0 0 {name=Hp vnam=vdp value=1}
            C {"""+rf"{self.symbol_dir}/{kwargs['modeln']}.sym"+r"""} 600 -340 2 1 {name=M1
            """ + f"""
            {kwargs['mn_supplement']}
            model={kwargs['modeln']}
            """ + """spiceprefix=X
            }
            C {"""+f"{self.symbol_dir}/{kwargs['modelp']}.sym"+r"""} 1600 -250 2 1 {name=M2
            """ + f"""
            {kwargs['mp_supplement']}
            model={kwargs['modelp']}
            """ + """spiceprefix=X
            }
            C {devices/code_shown.sym} 50 -10 0 0 {name=MODEL only_toplevel=true
            format="tcleval( @value )"
            value=".lib """ + kwargs['modelfile'] + """ mos_tt"}
            """

            with open(self.schematic_filepath, 'w') as f:
                f.write(multiline_join(schem))
            print("Generating netlist using xschem...")
            cmd_args = ['xschem', '--detach', '--netlist', self.schematic_filepath, '-o', str(Path(self.netlist_filepath).parent.resolve()), '--rcfile', os.path.expandvars(xschem_rc)]
        
            subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(self.netlist_filepath, 'r') as f:
                netlist = f.read()
            return netlist
        
        return multiline_join(f""" 
        ** sch_path: {self.schematic_filepath}
        **.subckt pysweep
        vgn gn 0 DC 0.6 AC 1
        vdn dn 0 0.6
        vbn bn 0 0
        vgp 0 gp DC 0.6 AC 1
        vdp 0 dp 0.6
        vbp 0 bp 0
        Hn nn 0 vdn 1
        Hp 0 np vdp 1
        XM1 0 gn dn bn {kwargs['modeln']}"""+""" w={wx} l={lx} ng=1 m=1
        """ + f"""XM2 0 gp dp bp {kwargs['modelp']}"""+""" w={wx} l={lx} ng=1 m=1
        **** begin user architecture code
        """ +"\n".join([f".lib {os.path.expandvars(kwargs['modelfile'])} mos_tt"] + sweep_codes) + """
        **** end user architecture code
        **.ends
        .end
        """)

    def run(self):
        Ls = self._config['SWEEP']['LENGTH']
        VSBs = self._config['SWEEP']['VSB']

        nch = self._config.generate_m_dict()
        pch = self._config.generate_m_dict()
        dimshape = (len(Ls),len(nch['VGS']),len(nch['VDS']),len(VSBs))
        for outvar in self._config['outvars']:
            nch[outvar] = np.zeros(dimshape, order='F')
            pch[outvar] = np.zeros(dimshape, order='F')

        for outvar in self._config['outvars_noise']:
            nch[outvar] = np.zeros(dimshape, order='F')
            pch[outvar] = np.zeros(dimshape, order='F')

        self._run_sim()
        n_dict, p_dict = self.extract_sweep_params(self.output)
        
        for n,p in zip(self._config['n'],self._config['p']):
            params_n = n
            values_n = n_dict[params_n[0]]
            params_p = p
            values_p = p_dict[params_p[0]]
            for m, outvar in enumerate(self._config['outvars']):
                nch[outvar] += np.squeeze(values_n*params_n[2][m])
                pch[outvar] += np.squeeze(values_p*params_p[2][m])

        for n,p in zip(self._config['n_noise'],self._config['p_noise']):
            params_n = n
            values_n = n_dict[params_n[0]]
            params_p = p
            values_p = p_dict[params_p[0]]
            for m, outvar in enumerate(self._config['outvars_noise']):
                nch[outvar] += np.squeeze(values_n)
                pch[outvar] += np.squeeze(values_p)
        return self._cleanup(nch, pch)
        
    def _run_sim(self):
        try:
            cmd_args = ['ngspice'] + self.args
            print(f"Running command: {' '.join(cmd_args)}")
            subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sleep(1)
        except subprocess.CalledProcessError as e:
            logging.info(f"Error executing process\n\n{e}")
        
    def _cleanup(self, nch, pch) -> Tuple[str, str]:
        clean_path = lambda ext: self.output.replace("txt", ext)
        for p in map(clean_path, ("txt", "log")):
            try:
                os.remove(p)
            except OSError as e:
                print(f"Could not perform cleanup:\nFile - {e.filename}\nError - {e.strerror}")

        for f in [self.netlist_filepath.replace(self.netlist_ext, "sch"), "pysweep.raw"]:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Could not perform cleanup:\nFile - {e.filename}\nError - {e.strerror}")

        return super()._cleanup(nch, pch)

    def extract_sweep_params(self, sweep_output_directory, sweep_type="DC"):
        df = pd.read_csv(sweep_output_directory, sep=r'\s+')
        df = df.apply(pd.to_numeric)

        df.columns = df.columns.str.replace('[dc]', '')
        df.columns = df.columns.str.replace('onoise.', '')
        df.columns = df.columns.str.replace('@', '')

        output_dicts = []
        for dev_name in ("MODELN", "MODELP"):
            dev_type = self._config['MODEL'][dev_name]
            # ngspice sweep order is l, vgs, vds, vsb
            dev_df = df.filter(regex=f'{dev_type}')
            dev_df.columns = dev_df.columns.str.replace(dev_type, '')
            dev_df.columns = dev_df.columns.str.replace('[', ':')
            dev_df.columns = dev_df.columns.str.replace(']', '')

            dev = dev_name[-1].lower()
            # l = np.unique(dev_df['l']) * 1e6    # convert to microns
            # assert np.all(np.isclose(l, self._config['SWEEP']['LENGTH'])), f"Length sweep values do not match configuration. (Expected {self._config['SWEEP']['LENGTH']}, got {l})"
            l = self._config['SWEEP']['LENGTH']
            vgs = np.unique(df[f'vg{dev}'])
            assert np.all(np.isclose(vgs, self._config['SWEEP']['VGS'])), f"VGS sweep values do not match configuration. (Expected {self._config['SWEEP']['VGS']}, got {vgs})"
            vds = np.unique(df[f'vd{dev}'])
            assert np.all(np.isclose(vds, self._config['SWEEP']['VDS'])), f"VDS sweep values do not match configuration. (Expected {self._config['SWEEP']['VDS']}, got {vds})"
            vsb = np.unique(-df[f'vb{dev}'])
            assert np.all(np.isclose(vsb, self._config['SWEEP']['VSB'])), f"VSB sweep values do not match configuration. (Expected {self._config['SWEEP']['VSB']}, got {vsb})"
            dims = [len(l), len(vgs), len(vds), len(vsb)]

            output_dicts.append(dict(map(lambda item: (item[0], np.reshape(item[1], dims)), dev_df.to_dict(orient='list').items())))

        return tuple(output_dicts)
