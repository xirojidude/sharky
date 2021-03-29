
Shader "Skybox/AsteroidAbduction"
{
    Properties
    {
        _MainTex1 ("tex2D", 2D) = "white" {}
        _MainTex2 ("tex2D", 2D) = "white" {}
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
// @author Jan <LJ> Scheurer


#define time _Time.y*.5

            uniform sampler2D _MainTex1; 
            uniform sampler2D _MainTex2; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

float3 ro,rs;
float rand(float2 p){return frac(sin(dot(float2(12.404,48.31243),p))*42634.653123);}
float rand(float3 p){return frac(sin(dot(float3(12.404,23.5213,48.31243),p))*42634.653123);}
const float2 O=float2(1,0);
float noise(float2 p){return tex2D(_MainTex1,p/64.).r;}
float noise(float3 p){
    float3 b=floor(p),f=frac(p);return lerp(
        lerp(lerp(rand(b+O.yyy),rand(b+O.xyy),f.x),lerp(rand(b+O.yxy),rand(b+O.xxy),f.x),f.y),
        lerp(lerp(rand(b+O.yyx),rand(b+O.xyx),f.x),lerp(rand(b+O.yxx),rand(b+O.xxx),f.x),f.y),f.z);
}
float gn(float2 p){return tex2D(_MainTex2,p/128.).r*1.2+tex2D(_MainTex1,p/32.).r*.1;}
float2x2 r2d(float a){float sa=sin(a),ca=cos(a);return float2x2(ca,sa,-sa,ca);}
float bmin(float a,float b,float k){return min(min(a,b),max(a,b)-k);}
float2 amod(float2 p,float a){a=(atan2(p.x,p.y))%a-a*.5;return float2(cos(a),sin(a))*length(p);}
float map(float3 p){
    float a=0.,cid=rand(floor(p.xz/20.+.5)*10.),d=p.y+gn(p.xz*.1)*4.-1.;
    if(d<.5)
        d-=max(tex2D(_MainTex2,p.xz*.05).g-.6,0.)*.2+
            max(6.-length(p),0.)+
            min(tex2D(_MainTex2,p.xz).b,.5)*
            smoothstep(2.,0.,length(ro-p))*.01+
            max(gn(p.xz)-.8,0.)*.5
        ;
    float3 o=p,q;
    p.xz=(p.xz+10.)%20.-10.;
    float2x2 r=r2d(cid*.7-.3);
    p.y-=1.;
    if(length(o.xz)<7.)p.y-=4.+sin(time*.2),a=1.;
    p.yz=mul(p.yz,r),p.xz=mul(p.xz,r);
    q=p;q.yz=mul(q.yz,r);
    p.xz=mul(p.xz,r2d(p.y*.2));
    p=abs(p)-(float2(2.-max(-p.y,0.)*.5,2.).xyx+float3(gn(p.zy*float2(.1,1)*1.5),gn(p.xz)*.5,gn(p.xy*float2(.1,1)*1.5)));
    d=min(min(d,-length(o)+60.),bmin(
        length(max(p+noise(p*4.)*.3,0.))-.2-a,
        max(max(abs(q.x)+2.,abs(q.z)-3.*cid)-3.,abs(q.y-2.)-.4),
        .5
    )-cid);
    return min(d,max(
        noise(       (o*5.+float3(0,-time*3.*sign(o.y-3.5),0))%100.      -50.)-.1-step(3.5,o.y)*.05,
        max(length(o.xz)-5.+noise(o.xy*.4)*2.,-o.y-5.))
    );
}
const float2 N=float2(.005,0);
float3 render(float3 rd){
    float md;float3 mp=ro;
    for(int i=0;i<70;i++)if(mp+=rd*(md=map(mp)),md<.001)break;
    float3
        n=normalize(map(mp)-float3(map(mp-N.xyy),map(mp-N.yxy),map(mp-N.yyx))),
        lp=float3(-5.,9.,2.),
        l=normalize(lp-mp)
    ;
    float
        ao=map(mp+n*.5)*.1+map(mp+n*2.)*.05,
        falloff=max(1.-length(lp-mp)*.07,0.)
    ;
    falloff*=falloff;
    return pow(tex2D(_MainTex2,mp.xz*.0005).rgb,float3(3.,3,3))+
        lerp(
            tex2D(_MainTex2,mp.xz*.2).rgb*max(dot(n,l),0.)*3.*.03+tex2D(_MainTex2,mp.xz).rgb*ao*.0375,
            float3(.2,.15,.1)*.4+gn(rd.xy*3.+time*.1)*.1,
            smoothstep(-5.,50.,length(ro-mp)-gn(rd.xy*5.+time*.1)*10.)
        )*5.
    ;
        
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex1, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

float2 p=float2(1,1);
float2 uv=float2(1,1);

//    float2 uv=fragCoord/iResolution.xy,p=uv*2.-1.;
//    p.x*=iResolution.x/iResolution.y;
//    ro=float3(-7.,-.5,13);
//    float3 rd=normalize(float3(p*r2d(sin(time*.12)*.1),-1.3+length(p)*.2));
    float2x2 rt=r2d(10.635+sin(time*.2)*.02);
    rd.yz=mul(rd.yx,r2d(-.35));
    ro.xz=mul(ro.xz,rt);
    rd.xz=mul(rd.xz,rt);
    fragColor=float4(0,0,0,0);
    #ifdef ANAGLYPH
    float focal=-.015;
    rs=rd;
    rs.xz=mul(rs.xz,r2d(focal));
    fragColor.rgb += render(rs)*float3(1,0,0);
    ro.xz+=mul(float2(.035,0),rt);
    rs.xz=mul(rd.xz,r2d(-focal));
    fragColor.rgb += render(rs)*float3(0,1,1);
    #else
    fragColor.rgb += render(rd);
    #endif
 //   fragColor.rgb = pow(fragColor.rgb,float3(1./2.2,1./2.2,1./2.2));
//    fragColor.rgb = mul(fragColor.rgb, 
//       float3(1.1,1.1,1.)*.9*max(1.-length(p)*.45+.7*uv.y*smoothstep(-.1,0.2,max(abs(p.x-.65-p.y*.2)-.32,p.y+p.x*.2-.5))*max(1.-abs(p.x-.8),0.)*2.*gn(p.xx*5.-p.y),0.)
//                    );
//    fragColor.rgb = 1.1*lerp(fragColor.rgb,float3(.43,.4,.4),(gn(p*3.+noise(float3(p,time))-time)*noise(p))*.5);


                return fragColor;
            }

            ENDCG
        }
    }
}




