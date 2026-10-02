"""
The symbolic regression engine behind :class:`~empulse.models.ProfSRClassifier`.

The expression representation, the genetic operators and the structure of the evolutionary loop are
adapted from gplearn 0.4.3 (https://github.com/trevorstephens/gplearn), whose license follows.
The fitness cache, the length limit, the Pareto front and the constant tuning are additions.

BSD 3-Clause License

Copyright (c) 2015-2026, Trevor Stephens
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of gplearn nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from ._evolve import EvolutionResult, Fitness, ParetoPoint, SearchSettings, evolve
from ._functions import FUNCTION_NAMES, Function, get_function
from ._program import Program, ProgramSpace

__all__ = [
    'FUNCTION_NAMES',
    'EvolutionResult',
    'Fitness',
    'Function',
    'ParetoPoint',
    'Program',
    'ProgramSpace',
    'SearchSettings',
    'evolve',
    'get_function',
]
