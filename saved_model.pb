ЩЛ
▄┐
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02unknown8╗я
x
dqn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedqn/dense_1/bias
q
$dqn/dense_1/bias/Read/ReadVariableOpReadVariableOpdqn/dense_1/bias*
_output_shapes
:*
dtype0
Б
dqn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*#
shared_namedqn/dense_1/kernel
z
&dqn/dense_1/kernel/Read/ReadVariableOpReadVariableOpdqn/dense_1/kernel*
_output_shapes
:	А*
dtype0
u
dqn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedqn/dense/bias
n
"dqn/dense/bias/Read/ReadVariableOpReadVariableOpdqn/dense/bias*
_output_shapes	
:А*
dtype0
~
dqn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А	А*!
shared_namedqn/dense/kernel
w
$dqn/dense/kernel/Read/ReadVariableOpReadVariableOpdqn/dense/kernel* 
_output_shapes
:
А	А*
dtype0
z
dqn/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namedqn/conv2d_2/bias
s
%dqn/conv2d_2/bias/Read/ReadVariableOpReadVariableOpdqn/conv2d_2/bias*
_output_shapes
:@*
dtype0
К
dqn/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_namedqn/conv2d_2/kernel
Г
'dqn/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpdqn/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
z
dqn/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namedqn/conv2d_1/bias
s
%dqn/conv2d_1/bias/Read/ReadVariableOpReadVariableOpdqn/conv2d_1/bias*
_output_shapes
:@*
dtype0
К
dqn/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_namedqn/conv2d_1/kernel
Г
'dqn/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpdqn/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
v
dqn/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namedqn/conv2d/bias
o
#dqn/conv2d/bias/Read/ReadVariableOpReadVariableOpdqn/conv2d/bias*
_output_shapes
: *
dtype0
Ж
dqn/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namedqn/conv2d/kernel

%dqn/conv2d/kernel/Read/ReadVariableOpReadVariableOpdqn/conv2d/kernel*&
_output_shapes
: *
dtype0
К
serving_default_input_1Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
 
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dqn/conv2d/kerneldqn/conv2d/biasdqn/conv2d_1/kerneldqn/conv2d_1/biasdqn/conv2d_2/kerneldqn/conv2d_2/biasdqn/dense/kerneldqn/dense/biasdqn/dense_1/kerneldqn/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_signature_wrapper_442

NoOpNoOp
╥(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Н(
valueГ(BА( B∙'
М
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2
	
conv3
flatten
fc1
fc2
	optimizer

signatures*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
░
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
 trace_1* 

!trace_0
"trace_1* 
* 
╚
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias
 )_jit_compiled_convolution_op*
╚
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias
 0_jit_compiled_convolution_op*
╚
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias
 7_jit_compiled_convolution_op*
О
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
ж
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

kernel
bias*
ж
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

kernel
bias*
* 

Jserving_default* 
QK
VARIABLE_VALUEdqn/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdqn/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdqn/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdqn/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdqn/conv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdqn/conv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdqn/dense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdqn/dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdqn/dense_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdqn/dense_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
	1

2
3
4
5*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
У
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Ptrace_0* 

Qtrace_0* 
* 

0
1*

0
1*
* 
У
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Wtrace_0* 

Xtrace_0* 
* 

0
1*

0
1*
* 
У
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
* 
* 
* 
* 
С
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

etrace_0* 

ftrace_0* 

0
1*

0
1*
* 
У
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

ltrace_0* 

mtrace_0* 

0
1*

0
1*
* 
У
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dqn/conv2d/kernel/Read/ReadVariableOp#dqn/conv2d/bias/Read/ReadVariableOp'dqn/conv2d_1/kernel/Read/ReadVariableOp%dqn/conv2d_1/bias/Read/ReadVariableOp'dqn/conv2d_2/kernel/Read/ReadVariableOp%dqn/conv2d_2/bias/Read/ReadVariableOp$dqn/dense/kernel/Read/ReadVariableOp"dqn/dense/bias/Read/ReadVariableOp&dqn/dense_1/kernel/Read/ReadVariableOp$dqn/dense_1/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *%
f R
__inference__traced_save_670
┘
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn/conv2d/kerneldqn/conv2d/biasdqn/conv2d_1/kerneldqn/conv2d_1/biasdqn/conv2d_2/kerneldqn/conv2d_2/biasdqn/dense/kerneldqn/dense/biasdqn/dense_1/kerneldqn/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_restore_710ад
╜
У
%__inference_dense_1_layer_call_fn_607

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_274o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б

Є
>__inference_dense_layer_call_and_return_conditional_losses_258

inputs2
matmul_readvariableop_resource:
А	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
А
·
A__inference_conv2d_2_layer_call_and_return_conditional_losses_567

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╘
┴
<__inference_dqn_layer_call_and_return_conditional_losses_415
input_1$

conv2d_388: 

conv2d_390: &
conv2d_1_393: @
conv2d_1_395:@&
conv2d_2_398:@@
conv2d_2_400:@
	dense_404:
А	А
	dense_406:	А
dense_1_409:	А
dense_1_411:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallш
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1
conv2d_388
conv2d_390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_199Р
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_393conv2d_1_395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_216Т
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_398conv2d_2_400*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_233╫
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_245Ў
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_404	dense_406*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_258Г
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_409dense_1_411*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_274w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
А
·
A__inference_conv2d_2_layer_call_and_return_conditional_losses_233

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┬
\
@__inference_flatten_layer_call_and_return_conditional_losses_245

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
з
A
%__inference_flatten_layer_call_fn_572

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_245a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╫.
э
<__inference_dqn_layer_call_and_return_conditional_losses_507

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
А	А4
%dense_biasadd_readvariableop_resource:	А9
&dense_1_matmul_readvariableop_resource:	А5
'dense_1_biasadd_readvariableop_resource:
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpК
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0з
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╛
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0└
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  В
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         А	В
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         АЕ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Й
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о

В
!__inference_dqn_layer_call_fn_467

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
А	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *E
f@R>
<__inference_dqn_layer_call_and_return_conditional_losses_281o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ы
█
__inference__traced_save_670
file_prefix0
,savev2_dqn_conv2d_kernel_read_readvariableop.
*savev2_dqn_conv2d_bias_read_readvariableop2
.savev2_dqn_conv2d_1_kernel_read_readvariableop0
,savev2_dqn_conv2d_1_bias_read_readvariableop2
.savev2_dqn_conv2d_2_kernel_read_readvariableop0
,savev2_dqn_conv2d_2_bias_read_readvariableop/
+savev2_dqn_dense_kernel_read_readvariableop-
)savev2_dqn_dense_bias_read_readvariableop1
-savev2_dqn_dense_1_kernel_read_readvariableop/
+savev2_dqn_dense_1_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ъ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├
value╣B╢B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B Д
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dqn_conv2d_kernel_read_readvariableop*savev2_dqn_conv2d_bias_read_readvariableop.savev2_dqn_conv2d_1_kernel_read_readvariableop,savev2_dqn_conv2d_1_bias_read_readvariableop.savev2_dqn_conv2d_2_kernel_read_readvariableop,savev2_dqn_conv2d_2_bias_read_readvariableop+savev2_dqn_dense_kernel_read_readvariableop)savev2_dqn_dense_bias_read_readvariableop-savev2_dqn_dense_1_kernel_read_readvariableop+savev2_dqn_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Г
_input_shapesr
p: : : : @:@:@@:@:
А	А:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
А	А:!

_output_shapes	
:А:%	!

_output_shapes
:	А: 


_output_shapes
::

_output_shapes
: 
ф
Ы
&__inference_conv2d_2_layer_call_fn_556

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_233w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╟	
Є
@__inference_dense_1_layer_call_and_return_conditional_losses_274

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б

Є
>__inference_dense_layer_call_and_return_conditional_losses_598

inputs2
matmul_readvariableop_resource:
А	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
У

Г
!__inference_signature_wrapper_442
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
А	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__wrapped_model_181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
¤1
а
__inference__wrapped_model_181
input_1C
)dqn_conv2d_conv2d_readvariableop_resource: 8
*dqn_conv2d_biasadd_readvariableop_resource: E
+dqn_conv2d_1_conv2d_readvariableop_resource: @:
,dqn_conv2d_1_biasadd_readvariableop_resource:@E
+dqn_conv2d_2_conv2d_readvariableop_resource:@@:
,dqn_conv2d_2_biasadd_readvariableop_resource:@<
(dqn_dense_matmul_readvariableop_resource:
А	А8
)dqn_dense_biasadd_readvariableop_resource:	А=
*dqn_dense_1_matmul_readvariableop_resource:	А9
+dqn_dense_1_biasadd_readvariableop_resource:
identityИв!dqn/conv2d/BiasAdd/ReadVariableOpв dqn/conv2d/Conv2D/ReadVariableOpв#dqn/conv2d_1/BiasAdd/ReadVariableOpв"dqn/conv2d_1/Conv2D/ReadVariableOpв#dqn/conv2d_2/BiasAdd/ReadVariableOpв"dqn/conv2d_2/Conv2D/ReadVariableOpв dqn/dense/BiasAdd/ReadVariableOpвdqn/dense/MatMul/ReadVariableOpв"dqn/dense_1/BiasAdd/ReadVariableOpв!dqn/dense_1/MatMul/ReadVariableOpТ
 dqn/conv2d/Conv2D/ReadVariableOpReadVariableOp)dqn_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0░
dqn/conv2d/Conv2DConv2Dinput_1(dqn/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
И
!dqn/conv2d/BiasAdd/ReadVariableOpReadVariableOp*dqn_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ю
dqn/conv2d/BiasAddBiasAdddqn/conv2d/Conv2D:output:0)dqn/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          n
dqn/conv2d/ReluReludqn/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          Ц
"dqn/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+dqn_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0╩
dqn/conv2d_1/Conv2DConv2Ddqn/conv2d/Relu:activations:0*dqn/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
М
#dqn/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,dqn_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
dqn/conv2d_1/BiasAddBiasAdddqn/conv2d_1/Conv2D:output:0+dqn/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @r
dqn/conv2d_1/ReluReludqn/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @Ц
"dqn/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+dqn_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╠
dqn/conv2d_2/Conv2DConv2Ddqn/conv2d_1/Relu:activations:0*dqn/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
М
#dqn/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,dqn_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0д
dqn/conv2d_2/BiasAddBiasAdddqn/conv2d_2/Conv2D:output:0+dqn/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @r
dqn/conv2d_2/ReluReludqn/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @b
dqn/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  О
dqn/flatten/ReshapeReshapedqn/conv2d_2/Relu:activations:0dqn/flatten/Const:output:0*
T0*(
_output_shapes
:         А	К
dqn/dense/MatMul/ReadVariableOpReadVariableOp(dqn_dense_matmul_readvariableop_resource* 
_output_shapes
:
А	А*
dtype0Ф
dqn/dense/MatMulMatMuldqn/flatten/Reshape:output:0'dqn/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЗ
 dqn/dense/BiasAdd/ReadVariableOpReadVariableOp)dqn_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
dqn/dense/BiasAddBiasAdddqn/dense/MatMul:product:0(dqn/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аe
dqn/dense/ReluReludqn/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         АН
!dqn/dense_1/MatMul/ReadVariableOpReadVariableOp*dqn_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ч
dqn/dense_1/MatMulMatMuldqn/dense/Relu:activations:0)dqn/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         К
"dqn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+dqn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
dqn/dense_1/BiasAddBiasAdddqn/dense_1/MatMul:product:0*dqn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         k
IdentityIdentitydqn/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ▒
NoOpNoOp"^dqn/conv2d/BiasAdd/ReadVariableOp!^dqn/conv2d/Conv2D/ReadVariableOp$^dqn/conv2d_1/BiasAdd/ReadVariableOp#^dqn/conv2d_1/Conv2D/ReadVariableOp$^dqn/conv2d_2/BiasAdd/ReadVariableOp#^dqn/conv2d_2/Conv2D/ReadVariableOp!^dqn/dense/BiasAdd/ReadVariableOp ^dqn/dense/MatMul/ReadVariableOp#^dqn/dense_1/BiasAdd/ReadVariableOp"^dqn/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2F
!dqn/conv2d/BiasAdd/ReadVariableOp!dqn/conv2d/BiasAdd/ReadVariableOp2D
 dqn/conv2d/Conv2D/ReadVariableOp dqn/conv2d/Conv2D/ReadVariableOp2J
#dqn/conv2d_1/BiasAdd/ReadVariableOp#dqn/conv2d_1/BiasAdd/ReadVariableOp2H
"dqn/conv2d_1/Conv2D/ReadVariableOp"dqn/conv2d_1/Conv2D/ReadVariableOp2J
#dqn/conv2d_2/BiasAdd/ReadVariableOp#dqn/conv2d_2/BiasAdd/ReadVariableOp2H
"dqn/conv2d_2/Conv2D/ReadVariableOp"dqn/conv2d_2/Conv2D/ReadVariableOp2D
 dqn/dense/BiasAdd/ReadVariableOp dqn/dense/BiasAdd/ReadVariableOp2B
dqn/dense/MatMul/ReadVariableOpdqn/dense/MatMul/ReadVariableOp2H
"dqn/dense_1/BiasAdd/ReadVariableOp"dqn/dense_1/BiasAdd/ReadVariableOp2F
!dqn/dense_1/MatMul/ReadVariableOp!dqn/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
А
·
A__inference_conv2d_1_layer_call_and_return_conditional_losses_547

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
р
Щ
$__inference_conv2d_layer_call_fn_516

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_199w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▒

Г
!__inference_dqn_layer_call_fn_304
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
А	А
	unknown_6:	А
	unknown_7:	А
	unknown_8:
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *E
f@R>
<__inference_dqn_layer_call_and_return_conditional_losses_281o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
ф
Ы
&__inference_conv2d_1_layer_call_fn_536

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_216w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╜
У
#__inference_dense_layer_call_fn_587

inputs
unknown:
А	А
	unknown_0:	А
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_258p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А	
 
_user_specified_nameinputs
┬
\
@__inference_flatten_layer_call_and_return_conditional_losses_578

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    А  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
А
·
A__inference_conv2d_1_layer_call_and_return_conditional_losses_216

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
р*
╩
__inference__traced_restore_710
file_prefix<
"assignvariableop_dqn_conv2d_kernel: 0
"assignvariableop_1_dqn_conv2d_bias: @
&assignvariableop_2_dqn_conv2d_1_kernel: @2
$assignvariableop_3_dqn_conv2d_1_bias:@@
&assignvariableop_4_dqn_conv2d_2_kernel:@@2
$assignvariableop_5_dqn_conv2d_2_bias:@7
#assignvariableop_6_dqn_dense_kernel:
А	А0
!assignvariableop_7_dqn_dense_bias:	А8
%assignvariableop_8_dqn_dense_1_kernel:	А1
#assignvariableop_9_dqn_dense_1_bias:
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Э
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├
value╣B╢B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ╒
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOpAssignVariableOp"assignvariableop_dqn_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dqn_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_2AssignVariableOp&assignvariableop_2_dqn_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_3AssignVariableOp$assignvariableop_3_dqn_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_4AssignVariableOp&assignvariableop_4_dqn_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_5AssignVariableOp$assignvariableop_5_dqn_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dqn_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dqn_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dqn_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dqn_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 л
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: Ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╤
└
<__inference_dqn_layer_call_and_return_conditional_losses_281

inputs$

conv2d_200: 

conv2d_202: &
conv2d_1_217: @
conv2d_1_219:@&
conv2d_2_234:@@
conv2d_2_236:@
	dense_259:
А	А
	dense_261:	А
dense_1_275:	А
dense_1_277:
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallч
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs
conv2d_200
conv2d_202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_199Р
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_217conv2d_1_219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_216Т
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_234conv2d_2_236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_233╫
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_245Ў
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	dense_259	dense_261*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_258Г
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_275dense_1_277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_274w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╟	
Є
@__inference_dense_1_layer_call_and_return_conditional_losses_617

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
■

°
?__inference_conv2d_layer_call_and_return_conditional_losses_527

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
■

°
?__inference_conv2d_layer_call_and_return_conditional_losses_199

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
C
input_18
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:ТИ
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2
	
conv3
flatten
fc1
fc2
	optimizer

signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
в
trace_0
 trace_12ы
!__inference_dqn_layer_call_fn_304
!__inference_dqn_layer_call_fn_467в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0z trace_1
╪
!trace_0
"trace_12б
<__inference_dqn_layer_call_and_return_conditional_losses_507
<__inference_dqn_layer_call_and_return_conditional_losses_415в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z!trace_0z"trace_1
╔B╞
__inference__wrapped_model_181input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▌
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

kernel
bias
 )_jit_compiled_convolution_op"
_tf_keras_layer
▌
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias
 0_jit_compiled_convolution_op"
_tf_keras_layer
▌
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias
 7_jit_compiled_convolution_op"
_tf_keras_layer
е
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
"
	optimizer
,
Jserving_default"
signature_map
+:) 2dqn/conv2d/kernel
: 2dqn/conv2d/bias
-:+ @2dqn/conv2d_1/kernel
:@2dqn/conv2d_1/bias
-:+@@2dqn/conv2d_2/kernel
:@2dqn/conv2d_2/bias
$:"
А	А2dqn/dense/kernel
:А2dqn/dense/bias
%:#	А2dqn/dense_1/kernel
:2dqn/dense_1/bias
 "
trackable_list_wrapper
J
0
	1

2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╓B╙
!__inference_dqn_layer_call_fn_304input_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒B╥
!__inference_dqn_layer_call_fn_467inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЁBэ
<__inference_dqn_layer_call_and_return_conditional_losses_507inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
<__inference_dqn_layer_call_and_return_conditional_losses_415input_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ш
Ptrace_02╦
$__inference_conv2d_layer_call_fn_516в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zPtrace_0
Г
Qtrace_02ц
?__inference_conv2d_layer_call_and_return_conditional_losses_527в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zQtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ъ
Wtrace_02═
&__inference_conv2d_1_layer_call_fn_536в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zWtrace_0
Е
Xtrace_02ш
A__inference_conv2d_1_layer_call_and_return_conditional_losses_547в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zXtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
ъ
^trace_02═
&__inference_conv2d_2_layer_call_fn_556в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z^trace_0
Е
_trace_02ш
A__inference_conv2d_2_layer_call_and_return_conditional_losses_567в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z_trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
щ
etrace_02╠
%__inference_flatten_layer_call_fn_572в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zetrace_0
Д
ftrace_02ч
@__inference_flatten_layer_call_and_return_conditional_losses_578в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zftrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
ч
ltrace_02╩
#__inference_dense_layer_call_fn_587в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zltrace_0
В
mtrace_02х
>__inference_dense_layer_call_and_return_conditional_losses_598в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zmtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
щ
strace_02╠
%__inference_dense_1_layer_call_fn_607в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zstrace_0
Д
ttrace_02ч
@__inference_dense_1_layer_call_and_return_conditional_losses_617в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zttrace_0
╚B┼
!__inference_signature_wrapper_442input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╪B╒
$__inference_conv2d_layer_call_fn_516inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
?__inference_conv2d_layer_call_and_return_conditional_losses_527inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_conv2d_1_layer_call_fn_536inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_conv2d_1_layer_call_and_return_conditional_losses_547inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_conv2d_2_layer_call_fn_556inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_conv2d_2_layer_call_and_return_conditional_losses_567inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_flatten_layer_call_fn_572inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_flatten_layer_call_and_return_conditional_losses_578inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╫B╘
#__inference_dense_layer_call_fn_587inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЄBя
>__inference_dense_layer_call_and_return_conditional_losses_598inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
%__inference_dense_1_layer_call_fn_607inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_dense_1_layer_call_and_return_conditional_losses_617inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Э
__inference__wrapped_model_181{
8в5
.в+
)К&
input_1         
к "3к0
.
output_1"К
output_1         ▒
A__inference_conv2d_1_layer_call_and_return_conditional_losses_547l7в4
-в*
(К%
inputs          
к "-в*
#К 
0         @
Ъ Й
&__inference_conv2d_1_layer_call_fn_536_7в4
-в*
(К%
inputs          
к " К         @▒
A__inference_conv2d_2_layer_call_and_return_conditional_losses_567l7в4
-в*
(К%
inputs         @
к "-в*
#К 
0         @
Ъ Й
&__inference_conv2d_2_layer_call_fn_556_7в4
-в*
(К%
inputs         @
к " К         @п
?__inference_conv2d_layer_call_and_return_conditional_losses_527l7в4
-в*
(К%
inputs         
к "-в*
#К 
0          
Ъ З
$__inference_conv2d_layer_call_fn_516_7в4
-в*
(К%
inputs         
к " К          б
@__inference_dense_1_layer_call_and_return_conditional_losses_617]0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ y
%__inference_dense_1_layer_call_fn_607P0в-
&в#
!К
inputs         А
к "К         а
>__inference_dense_layer_call_and_return_conditional_losses_598^0в-
&в#
!К
inputs         А	
к "&в#
К
0         А
Ъ x
#__inference_dense_layer_call_fn_587Q0в-
&в#
!К
inputs         А	
к "К         Ан
<__inference_dqn_layer_call_and_return_conditional_losses_415m
8в5
.в+
)К&
input_1         
к "%в"
К
0         
Ъ м
<__inference_dqn_layer_call_and_return_conditional_losses_507l
7в4
-в*
(К%
inputs         
к "%в"
К
0         
Ъ Е
!__inference_dqn_layer_call_fn_304`
8в5
.в+
)К&
input_1         
к "К         Д
!__inference_dqn_layer_call_fn_467_
7в4
-в*
(К%
inputs         
к "К         е
@__inference_flatten_layer_call_and_return_conditional_losses_578a7в4
-в*
(К%
inputs         @
к "&в#
К
0         А	
Ъ }
%__inference_flatten_layer_call_fn_572T7в4
-в*
(К%
inputs         @
к "К         А	м
!__inference_signature_wrapper_442Ж
Cв@
в 
9к6
4
input_1)К&
input_1         "3к0
.
output_1"К
output_1         