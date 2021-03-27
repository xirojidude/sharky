Shader "Unlit/GalaxyNavigator"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);
                // apply fog
                UNITY_APPLY_FOG(i.fogCoord, col);
                return col;
            }


#define get(i)texture(iChannel0,vec2(i+.5,.5)/iChannelResolution[0].xy,-100.0)

//abriceNeyret2 Black Body...  https://www.shadertoy.com/view/4tdGWM
vec3 blackBody(float k
){float T=(k*2.)*16000.
 ;vec3 c=vec3(1.,3.375,8.)/(exp((19e3*vec3(1.,1.5,2.)/T))- 1.)
 ;return c/max(c.r,max(c.g,c.b));}

#define noiseTextureSize 256.

// iq noise https://www.shadertoy.com/view/4sfGzS
float noise(vec3 x
){vec3 p=floor(x),f=fract(x)
 ;f*=f*(3.-f-f)
 ;vec2 uv=(p.xy+vec2(37.,17.)*p.z)+ f.xy
 ,rg=textureLod(iChannel1,(uv+.5)/noiseTextureSize,-100.).yx
 ;return mix(rg.x,rg.y,f.z);}
float pn(vec3 x
){vec3 p=floor(x),f=fract(x)
 ;f*=f*(3.-f-f)
 ;vec2 uv=(p.xy+vec2(37.,17.)*p.z)+ f.xy
 ,rg=textureLod(iChannel1,(uv+.5)/noiseTextureSize,-100.).yx
 ;return 2.4*mix(rg.x,rg.y,f.z)-1.;}
float bm(vec3 x
){vec3 p=floor(x),f=fract(x)
 ;f*=f*(3.-f-f)
 ;vec2 uv=(p.xy+vec2(37.,17.)*p.z)+ f.xy
 ,rg=textureLod(iChannel1,(uv+ .5)/noiseTextureSize,-100.).yx
 ;return 1.-.82*mix(rg.x,rg.y,f.z);}
float fpn(vec3 p){return pn(p*.06125)*.5+pn(p*.125)*.25+pn(p*.25)*.125 ;}//+pn(p*.5)*.625
float fbm(const in vec3 p){return bm(p*.06125)*.5+bm(p*.125)*.25+bm(p*.25)*.125+bm(p*.4)*.2;}
float smoothNoise(in vec3 q
){const mat3 msun=mat3(0.,.8,.6,-.8,.36,-.48,-.6,-.48,.64)
 ;float f=.5000*noise(q);q=msun*q*2.01
 ;f+=.2500*noise(q);q=msun*q*2.02
 ;f+=.1250*noise(q);q=msun*q*2.03
 ;f+=.0625*noise(q)
 ;return f;}

// otaviogood's noise from https://www.shadertoy.com/view/ld2SzK
float SpiralNoiseC(vec3 p,vec4 id
){const float m=20.,n=inversesqrt(1.+m*m)
 ;float iter=2.,r=2.-id.x
 ;for (int i=0;i<SPIRAL_NOISE_ITER;i++
 ){r+=-abs(sin(p.y*iter)+ cos(p.x*iter))/iter
  ;p.xy+=vec2(p.y,-p.x)*m;p.xy*=n
  ;p.xz+=vec2(p.z,-p.x)*m;p.xz*=n
  ;iter*=id.y+.733733;}return r;}

float mapIntergalacticCloud(vec3 p,vec4 id
){float k=2.*id.w+.1;// p/=k
 ;return k*(.5+SpiralNoiseC(p.zxy*.4132+333.,id)*3.+pn(p*8.5)*.12);}

#ifdef WITH_SUPERNOVA_REMNANT

//Intersection functions (mainly iq)
//return bool and 2 intersection distances
bool traceSphere(v22 r,out float a,out float b//ray,near,far
){float c=dot(r.b,r.a);b=c*c-dot(r.a,r.a)+8.
 ;if(b<0.0)return false
 ;b=sqrt(b);a=-c-b;b-=c;return b>0.;}
//return bool,farIntersection and [edge]
bool traceSphere(v22 r,float s,out vec2 v // b,out float e
){v.x=dot(r.b,-r.a);v.y=v.x*v.x-dot(r.a,r.a)+s*s
 ;if(v.y<0.)return false
 ;v.y=sqrt(v.x);v.x=v.x-v.y;return v.x>0.;}
 
float SpiralNoiseC2(vec3 p
){const float m=.9;// size of perpendicular vector
 ;float n=inversesqrt(1.+m*m)
 ;float r=0.,iter=2.
 ;for (int i=0;i<8;i++
 ){r+=-abs(sin(p.y*iter)+ cos(p.x*iter))/ iter //abs is optional for rigged look
  ;p.xy+=vec2(p.y,-p.x)*m;p.xy*=n
  ;p.xz+=vec2(p.z,-p.x)*m;p.xz*=n
  ;iter*=1.733733;}return r;}
float length2(vec2 p){return sqrt(p.x*p.x+p.y*p.y);}
float length8(vec2 p){p=p*p;p=p*p;p=p*p;return pow(p.x+p.y,.125);}
float Disk(vec3 p,vec3 t
){vec2 q=vec2(length2(p.xy)-t.x,p.z*0.5)
 ;return max(length8(q)-t.y,abs(p.z)- t.z);}
float mapSupernovaRemnant(vec3 p
){p*=2.
 ;float noi=Disk(p.xzy,vec3(2.0,1.8,1.25))+fbm(p*90.)+SpiralNoiseC2(p.zxy*0.5123+100.0)*3.
 ;return abs(noi*.5)+.07;}
#endif // WITH_SUPERNOVA_REMNANT

bool cylinder(vec3 ro,vec3 rd,float r,float h,out float tn,out float tf
){float a=dot(rd.xy,rd.xy),b=dot(ro.xy,rd.xy)
 ,d=b*b- a*(dot(ro.xy,ro.xy)- r*r)
 ;if(d<0.)return false
 ;d=sqrt(d)
 ;tn=(-b- d)/a;tf=(-b+d)/a
 ;a=min(tf,tn);tf=max(tf,tn);tn=a// order roots
 ;a=ro.z+tn*rd.z
 ;b=ro.z+tf*rd.z
 ;vec2 zcap=h*vec2(.5,-.5),cap=(zcap- ro.z)/ rd.z
 ;tn=a<zcap.y?cap.y : a>zcap.x?cap.x : tn
 ;tf=b<zcap.y?cap.y : b>zcap.x?cap.x : tf
 ;return tf>0. && tf>tn;}

//Awesome star by Foxes: https://www.shadertoy.com/view/4lfSzS
float noise4q(vec4 x
){vec4 n3=vec4(0,.25,.5,.75)
 ;vec4 p2=floor(x.wwww+n3)
 ;vec4 b=floor(x.xxxx+n3)+ floor(x.yyyy+n3)*157.+floor(x.zzzz+n3)*113.
 ;vec4 p1=b+fract(p2*.00390625)*vec4(164352.,-164352.,163840.,-163840.)
 ;p2=b+fract((p2+1.0)*.00390625)*vec4(164352.,-164352.,163840.,-163840.)
 ;vec4 f1=fract(x.xxxx+n3),f2=fract(x.yyyy+n3)
 ;f1*=f1*(3.0-f1-f1)
 ;f2*=f2*(3.0-f2-f2)
 ;vec4 n1=vec4(0,1.,157.,158.),n2=vec4(113.,114.,270.0,271.)
 ;f1=fract(x.zzzz+n3)
 ;f2=fract(x.wwww+n3)
 ;f1*=f1*(3.-2.*f1)
 ;f2*=f2*(3.-2.*f2)
 ;vec4 vs1=mix(mix(mix(h3(p1),h3(n1.yyyy+p1)    ,f1)
          ,mix(h3(n1.zzzz+p1),h3(n1.wwww+p1)    ,f1),f2)
          ,mix(mix(h3(n2.xxxx+p1),h3(n2.yyyy+p1),f1)
          ,mix(h3(n2.zzzz+p1),h3(n2.wwww+p1)    ,f1),f2),f1)
 ;vec4 vs3=mix(mix(mix(h3(p2),h3(n1.yyyy+p2)    ,f1)
          ,mix(h3(n1.zzzz+p2),h3(n1.wwww+p2)    ,f1),f2)
          ,mix(mix(h3(n2.xxxx+p2),h3(n2.yyyy+p2),f1)
          ,mix(h3(n2.zzzz+p2),h3(n2.wwww+p2)    ,f1),f2),f1)
 ;vs1=mix(vs1,vs3,f2)
 ;float r=dot(vs1,vec4(.25))
 ;return r*r*(3.-r-r);}

// rays of a star
float ringRayNoise(v22 r,float s,float size,float anim
){float b=dot(r.b,r.a)
 ;vec3 pr=r.b*b-r.a
 ;float c=length(pr)
 ,m=max(0.,(1.-size*abs(s-c)))
 ;pr=pr/c
 ;float n=.4,ns=1.,nd=noise4q(vec4(pr*1.0,-anim+c))*2.
 ;if (c>s
 ){n=noise4q(vec4(pr*10.0,-anim+c))
  ;ns=noise4q(vec4(pr*50.0,-anim*2.5+ c+c))*2.;}
 ;return m*m*(m*m+n*n*nd*nd*ns);}

//Sun Lava effect

vec3 getSunColor(vec3 p,vec4 id,float time
){float lava=smoothNoise((p+vec3(time*.03))*50.*(.5+id.z))
 ;return blackBody(.02+3.*clamp(id.x*id.x,.05,1.)*(1.- sqrt(lava)));}

vec4 renderSun(v22 r,in vec4 i,in float t//ray,id,time
){r.a*=2.
 ;// Rotate view to integrate sun rotation 
 ;// R(ro.zx,1.6-t*.5*i.w)
 ;// R(rd.zx,1.6-t*.5*i.w)
 ;vec4 c=vec4(0)
 ;if(traceSphere(r,1.,c.xy))c=vec4(getSunColor(r.a+r.b*c.x,i,t),smoothstep(0.,.2,c.y));
 ;r.a.x=ringRayNoise(r,1.0,5.-4.*i.y,t)
 ;c.a=max(c.a,clamp(r.a.x,0.,.98))
 ;c.rgb+=blackBody(i.x)*r.a.x
 ;c.rgb*=1.-.03*cos(5.*t+2.*hash(t))// twinkle
 ;return sat(c);}

// Supernova remnant by Duke [https://www.shadertoy.com/view/MdKXzc]

#ifdef WITH_SUPERNOVA_REMNANT

vec3 computeColorSR(float d,float r//density,radius
){return mix(vec3(1.,.9,.8),vec3(.4,.15,.1),d)
 *mix(7.*vec3(.8,1.,1.),1.5*vec3(.48,0.53,.5),min((r+.5)/.9,1.15));}

vec4 renderSupernova(v22 r
){vec4 a=vec4(0.)//acc return
 ;float m=0.,n=0.
 ;if(traceSphere(r,m,n)
 ){float t=max(m,0.)+.01*hash(r.b)//calculate once
  ;r.a*=3.
  ;m=0.//acc conditional
  ;for(int i=0;i<64;i++
  ){if (m>.9 || a.w>.99 || t>n)break
   ;vec3 pos=r.a+r.b*t
   ;float d=mapSupernovaRemnant(pos)
   ;float lDist=max(length(pos),.001)
   ;a+=vec4(.67,.75,1.,1.)/(lDist*lDist*10.)*.0125// star 
   ;a+=vec4(1.,.5,.25,.6)/exp(lDist*lDist*lDist*.08)*.033// bloom
   ;const float h=.1
   ;if(d<h
   ){m+=(1.-m)*(h-d)+.005
    ;vec4 c=vec4(computeColorSR(m,lDist),m*.2)
    ;a.rgb+=a.w*a.rgb*.2;c.rgb*=c.a;a+=c*(1.- a.w);}
   ;m+=.014
   ;// trying to optimize step size near the camera and near the light source
   ;t+=max(d*.1*max(min(lDist,length(r.a)),1.0),0.01);}
  ;//a*=1./exp(l*.2)*.6; //scatter
  ;a=sat(a)
  ;a.xyz*=a.xyz*(3.-a.xyz-a.xyz);}return a;}
#endif

//Galaxy

//p,1/thickness,blurAmount,blurStyle
float spiralArm(vec3 p,float t,float a,float s
){float r=length(p.xz)
 ,l=log(r)
 ,d=(.5/2.- abs(fract(.5*(atan(p.x,p.z)-l*4.)/pi)-.5))*2.*pi*r*(1.-l)*t
 ;return sqrt(d*d+10.*p.y*p.y*t)-(.1-.25*r)*r*.2
 -a*mix(fpn(8.*vec3(r*43.,40.*d,24.*p.y)),fpn(p*400.),s)// Perturb
 ;}

void galaxyTransForm(inout vec3 o,vec4 i
){R(o.yz,(i.y-.5))
 ;// R(o.xy,.25*i.x*iTime);
 ;}

float mapGalaxy(vec3 p,vec4 id
){float t=.2/(1.+id.x)
 ;float d1=spiralArm(p.xzy*.2,t,.2+.3*id.y,id.z)
 #ifdef WITH_DOUBLE_GALAXY
 ;if(id.z<.25
 ){float d2=spiralArm(vec3(-p.y,p.z,p.x)*.205,t,.2+.3*id.y,id.z)
  ;return min(d2,d1);}
 #endif 
 ;return d1;}

vec3 computeColor(float d,vec3 u//density,uv
){u.x=length(u);return mix(vec3(.25,.22,.2),vec3(.1,.0375,.025),d)
 *mix(vec3(4.8,6.,6.), vec3(.96,1.06,1.),min((u.x+.5)*.5,1.15));}

vec4 renderGalaxy(v22 r,in vec4 id,in bool fast
){vec4 a=vec4(0)
 ;float min_dist=0.,max_dist=100.
 ;galaxyTransForm(r.a,id)
 ;galaxyTransForm(r.b,id)
 ;if(cylinder(r.a,r.b,3.,3.5,min_dist,max_dist)
 ){float b=0.//acc
  ,l=max(min_dist,0.)+ .2*hash(r.b+iTime)//acc raylength
  ;vec4 n,m//changes wiothin loop (composite vector)
  ;for (int i=0;i<48;i++ //raymarch
  ){if((fast&&i>20)|| b>.9 || a.w>0.99 || l>max_dist)break
   ;vec3 u=r.a+r.b*l
   ;float d=max(abs(mapGalaxy(3.5*u,id))+.05,.005)
   ;const float h=.1
   ;if(d<h
   ){float l=h-d
    ;l+=sat((l-mapGalaxy(u*3.5-.2*normalize(u),id))*2.5)
    ;b+=(1.-b)*l+.005
    ;vec4 c=vec4(computeColor(b,u),b*.25);c.rgb*=c.a;a+=c*(1.- a.w);}
   ;b+=.014
   ;n.xyz=u*.25
   ;m.xyz=u*.05
   ;m.z*=2.5
   ;n.w=max(length(n.xyz),.0001)//max(length(n.xyz),0.001)
   ;m.w=max(length(m.xyz),.0001)
   ;vec3 lightColor=(1.-smoothstep(3.,4.5,n.w*n.w))
   *mix(.07*vec3(1.,.5,.25)/n.w,.008*vec3(1.,1.7,2.)/m.w,smoothstep(.2,.7,n.w));// star in center
   ;a.rgb+=lightColor/(n.w*20.);//bloom
   ;d=max(d,.04)
   ;l+=max(d*.3,.02);}
  ;a=sat(a)
  ;a.xyz*=a.xyz*(3.-a.xyz-a.xyz);}return a;}

// Adapted from Planet Shadertoy- Reinder Nijhoff: https://www.shadertoy.com/view/4tjGRh
vec3 renderGalaxyField(v22 r,bool fast
){//out_id=vec3(9)
 ;float dint //changes within loop within point()
 ,l=0.       //acc rayLength
 ;vec3 h     //hash sepends on (pos) within loop
 ,o=r.a      //acc origin
 ,f=floor(o) //acc tileId pos
 ,v=1./r.b   //calculate once
 ,s=sign(r.b)//calculate once
 ,dis=(f-o+.5+s*.5)*v
 ;vec4 a=vec4(0)
 ;for(int i=0;i<GALAXY_FIELD_VOXEL_STEPS_HD;i++
 ){if(!fast||i!=0
  ){h=hash33(f)
   ;vec3 O=f+cl1(h,GALAXY_RADIUS)
   ;l=point(o,r.b,O,dint)
   ;if(dint>0. && l<GALAXY_RADIUS
   ){vec4 c=renderGalaxy(v22((o-O)/GALAXY_RADIUS*3.,r.b),vec4(h,.5),fast)
    ;c.rgb*=smoothstep(float(GALAXY_FIELD_VOXEL_STEPS),0.,length(r.a-f))
    ;a+=(1.-a.w)*c
    ;if(a.w>.99)break;}}
  ;vec3 m=step(dis.xyz,dis.yxy)* step(dis.xyz,dis.zzx)
  ;dis+=m*s*v;f+=m*s;}
 ;if(!fast && a.w<.99
 ){for(int i=GALAXY_FIELD_VOXEL_STEPS_HD;i<GALAXY_FIELD_VOXEL_STEPS;i++
  ){h=hash33(f)
   ;l=point(o,r.b,f+cl1(h,GALAXY_RADIUS),dint)
   ;if(dint>0.
   ){vec4 c=vec4(.9,.9,.8,1.)*(1.-smoothstep(GALAXY_RADIUS*.25,GALAXY_RADIUS*.5,l))
    ;c.rgb*=smoothstep(float(GALAXY_FIELD_VOXEL_STEPS),0.,length(r.a-f))
    ;a+=(1.-a.w)*c
    ;if(a.w>.99)break;}
   ;vec3 m=step(dis.xyz,dis.yxy)* step(dis.xyz,dis.zzx)
   ;dis+=m*s*v;f+=m*s;}}
 ;return a.xyz;}

// Adapted from Planet Shadertoy- Reinder Nijhoff:  https://www.shadertoy.com/view/4tjGRh
vec4 renderStarField(v22 r,inout float O
){float dint   //changes within loop within point()
 ,l=0.         //acc rayLength
 ;vec3 n=1./r.b//calculate once
 ,rs=sign(r.b) //calculate once
 ,o            //changes within loop (offset)
 ,h            //changes within loop (hash)
 ,f=floor(r.a) //changes within loop (tileId)
 ,v=(f-r.a+.5+rs*.5)*n//acc voxel coods
 ;vec4 c              //acc color intermediate
 ,a=vec4(0)           //acc color return
 ;for(int i=0;i<STAR_FIELD_VOXEL_STEPS;i++
 ){h=hash33(f)
  ;o=cl1(h,STAR_RADIUS)
  ;l=point(r.a,r.b,f+o,dint)
  ;if(dint>0.
  ){if(dint<2. && l<STAR_RADIUS
   ){r.a=(r.a-f-o)/STAR_RADIUS,
    #ifdef WITH_SUPERNOVA_REMNANT
     c=h.x>.8?renderSupernova(v22(r.a,r.b)):
    #endif
     c=renderSun(v22(r.a,r.b),vec4(h,.5),iTime)
    ;if (c.a>.99)O=dint
   ;}else c=(vec4(blackBody(max(h.x-.1,.01)),1.)*(1.-smoothstep(STAR_RADIUS*.5,STAR_RADIUS,l)))
   ;c.rgb*=smoothstep(float(STAR_FIELD_VOXEL_STEPS),.5,dint)
   ;c.rgb*=c.a
   ;a+=(1.-a.w)*c
   ;if (a.w>.99)break;}
  ;vec3 mm=step(v.xyz,v.yxy)* step(v.xyz,v.zzx)
  ;v+=mm*rs*n
  ;f+=mm*rs;}
 return a;}


//intergalactic clouds

#ifdef WITH_INTERGALACTIC_CLOUDS


// Based on "Type 2 Supernova" by Duke (https://www.shadertoy.com/view/lsyXDK)
vec4 renderIntergalacticClouds(v22 r,float m
){m=min(m,float(STAR_FIELD_VOXEL_STEPS)) //calculated once
 ;vec4 a=vec4(0),id=vec4(.5,.4,.16,.7)         //loop var
 ;vec3 u              //loop var
 ;float e=0.          //loop var       , edge color
 ,o=.05+.25*id.z      //calculated once, outer edge?
 ,l=hash(hash(r.b))*.1//calculated once, hashed rayLength
 ,b=smoothstep(m,0.,l)//calculated once
 ;for(int i=0;i<100;i++
 ){if(e>.9 || a.w>.99 || l>m)break
  ;u=r.a+l*r.b
  ;float d=abs(mapIntergalacticCloud(u,id))+.07//depends on var u
  ,sp=4.5//quickly discarded scalar constant
  ;sp=max(length(mod(u+sp,sp*2.)-sp),.001)
  ;u.x=pn(.05*u)
  ;vec3 c=mix(hsv2rgb(u.x,.5,.6),hsv2rgb(u.x+.3,.5,.6),smoothstep(2.*id.x*.5,2.*id.x*2.,sp))
  ;a.rgb+=b*c/exp(sp*sp*sp*.08)/30.
  ;if (d<o//color edges
  ){e+=(1.-e)*(o-d)+.005
   ;a.rgb+=a.w*a.rgb*.25/sp//emmissive
   ;a+=(1.-a.w)*.02*e*b; }// uniform scale density+alpha blend in contribution 
  ;e+=.015
  ;l+=max(d*.08*max(min(sp,d),2.),.01);}// trying to optimize step size
 ;a=sat(a)
 ;a.xyz*=a.xyz*(3.-a.xyz-a.xyz);return a;}

#endif 

// Coordinate system conversions
bool inGalaxy(vec3 u){vec3 f=floor(u)
 ;return length(u-f-cl1(hash33(f),GALAXY_RADIUS))<GALAXY_RADIUS;}

v22 getCam(vec2 p
){vec3 w=get(Bc1).xyz//is normalized on set()
 ,u=normalize(cross(w,normalize(vec3(.1*cos(.1*iTime),.3*cos(.1*iTime),1.))))
 ;return v22(get(Bc0).xyz,normalize(-p.x*u+p.y*normalize(cross(u,w))+2.*w) );}

void mainImage(out vec4 o,in vec2 u
){o.xyz=vec3(0)
 ;u=u/iResolution.xy
 ;u=u*2.-1.
 ;u.x*=iResolution.x/iResolution.y
 ;v22 r=getCam(u)
 ;vec3 galaxyId,galaxyPosU
 ;vec4 s=vec4(0)//star color
 ;float coo=get(Bvv).z
 ;coo=coo>2.5?IN_SOLAR_SYSTEM : coo>1.5?IN_GALAXY :  IN_UNIVERSE
 ;bool isU=coo==IN_UNIVERSE,isG=coo==IN_GALAXY
 ;vec3 galaxy_pos=get(Bg0).xyz
 ;if(inGalaxy(isU?r.a : g2u(galaxy_pos,r.a))
 ){vec3 roG=isU?u2g(galaxy_pos,r.a): r.a
  ;float d=9999.
  ;s=renderStarField(v22(roG,r.b),d)
  #ifdef WITH_INTERGALACTIC_CLOUDS
  ;vec4 c=renderIntergalacticClouds(v22(roG,r.b),d)
  ;if(s.a!=0.)s=(1.-c.a)*sqrt(s)*s.a
  ;s+=c
  #endif 
  ;}
 ;if(isG)r.a=g2u(galaxy_pos,r.a)
 ;vec3 colGalaxy=renderGalaxyField(r,isG)
 ;s.rgb+=colGalaxy*(1.-s.a)
 ;o.xyz=s.rgb
 ;// float digit=PrintValue(fragCoord,iResolution.xy*vec2(.0,.7),vec2(20.),galaxy_pos.x,8.,10.);
 ;// digit+=PrintValue(fragCoord,iResolution.xy*vec2(.0,.6),vec2(20.),galaxy_pos.y,8.,10.);
 ;// digit+=PrintValue(fragCoord,iResolution.xy*vec2(.0,.5),vec2(20.),galaxy_pos.z,8.,10.);
 ;// o.xyz=mix(o.xyz,vec3(1,0,0),digit)
 ;if(isU)o.xyz+=vec3(0.03,0.,.1);}



            ENDCG
        }
    }
}
