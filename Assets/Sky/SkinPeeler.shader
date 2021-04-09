
Shader "Skybox/SkinPeeler"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
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

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }

// Skin peeler
// by Dave Hoskins
// Originally from Xyptonjtroz by nimitz (twitter: @stormoid)
// Edited by Dave Hoskins, by changing atmosphere with quite a few lighting changes & added audio

#define ITR 100
#define FAR 30.
#define time _Time.y
float3 ligt;

/*
    Believable animated volumetric dust storm in 7 samples,
    blending each layer in based on geometry distance allows to
    render it without visible seams. 3d Triangle noise is 
    used for the dust volume.

    Further explanation of the dust generation...
        
    The basic idea is to have layers of gradient shaded volumetric
    animated noise. The problem is when geometry is intersected
    before the ray reaches the far plane. A way to smoothly blend
    the low sampled noise is needed.  So I am blending (smoothstep)
    each dust layer based on current ray distance and the solid 
    interesction distance. I am also scaling the noise taps as a 
    function of the current distance so that the distant dust doesn't
    appear too noisy and as a function of current height to get some
    "ground hugging" effect.
    
*/

#define MOD3 float3(.16532,.17369,.15787)
float2x2 mm2(in float a){float c = cos(a), s = sin(a);return float2x2(c,s,-s,c);}

float height(in float2 p)
{
    p *= 0.2;
    return sin(p.y)*0.4 + sin(p.x)*0.4;
}

//smooth min form iq
float smin( float a, float b)
{
    const float k = 0.7;
    float h = clamp( 0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
    return lerp( b, a, h ) - k*h*(1.0-h);
}

///  2 out, 2 in...
float2 hash22(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * MOD3);
    p3 += dot(p3.zxy, p3.yxz+19.19);
    return frac(float2(p3.x * p3.y, p3.z*p3.x));
}

float mod(float a, float b) {return a%b;}

float vine(float3 p, in float c, in float h)
{
    p.y += sin(p.z*.5625+1.3)*1.5-.5;
    p.x += cos(p.z*.4575)*1.;
    //p.z -=.3;
    float2 q = float2(mod(p.x, c)-c/2., p.y);
    return length(q) - h -sin(p.z*3.+sin(p.x*7.)*0.5+time)*0.13;
}

float map(float3 p)
{
    p.y += height(p.zx);
    
    float3 bp = p;
    float2 hs = hash22(floor(p.zx/4.));
    p.zx = mod(p.zx,4.)-2.;
    
    float d = p.y+0.5;
    p.y -= hs.x*0.4-0.15;
    p.zx += hs*1.3;
    d = smin(d, length(p)-hs.x*0.4);
    
    d = smin(d, vine(bp+float3(1.8,0.,0),15.,.8) );
    d = smin(d, vine(bp.zyx+float3(0.,0,17.),20.,0.75) );
    
    return d;
}
float shadow(in float3 ro, in float3 rd, in float mint, in float tmax)
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<20; i++ )
    {
        float h = map(ro + rd*t);
        res = min( res, 4.*h/t );
        t += clamp( h, 0.2, 1.5 );
            }
    return clamp( res, 0.0, 1.0 );

}

float march(in float3 ro, in float3 rd)
{
    float precis = 0.002;
    float h=precis*2.0;
    float d = 0.;
    for( int i=0; i<ITR; i++ )
    {
        if( abs(h)<precis || d>FAR ) break;
        d += h;
        h = map(ro+rd*d);
        
    }
    return d;
}

float tri(in float x){return abs(frac(x)-.5);}
float3 tri3(in float3 p){return float3( tri(p.z+tri(p.y*1.)), tri(p.z+tri(p.x*1.)), tri(p.y+tri(p.x*1.)));}
                                 
float2x2 m2 = float2x2(0.970,  0.242, -0.242,  0.970);

float triNoise3d(in float3 p)
{
    float z=1.4;
    float rz = 0.;
    float3 bp = p;
    for (float i=0.; i<=3.; i++ )
    {
        float3 dg = tri3(bp);
        p += (dg);

        bp *= 2.;
        z *= 1.5;
        p *= 1.2;
        //p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
    }
    return rz;
}

float fogmap(in float3 p, in float d)
{
   p.x += time;
   p.z += time*.5;
    
    return triNoise3d(p*2.2/(d+8.0))*(smoothstep(.7,.0,p.y));
}

float3 fog(in float3 col, in float3 ro, in float3 rd, in float mt)
{
    float d = .5;
    for(int i=0; i<7; i++)
    {
        float3  pos = ro + rd*d;
        float rz = fogmap(pos, d);
        float sh = shadow(pos, ligt, 1.1, 20.0);
        col = lerp(col,float3(.85, .65, .5),clamp(rz*smoothstep(d,d*1.8,mt),0.,1.) *sh);
        d *= 1.8;
        if (d>mt)break;
    }
    return col;
}

float3 normal(in float3 p)
{  
    float2 e = float2(-1., 1.)*0.005;   
    return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
                     e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}

float bnoise(in float3 p)
{
    float n = sin(triNoise3d(p*3.)*7.)*0.4;
    n += sin(triNoise3d(p*1.5)*7.)*0.2;
    return (n*n)*0.01;
}

float3 bump(in float3 p, in float3 n, in float ds)
{
    float2 e = float2(.005,0);
    float n0 = bnoise(p);
    float3 d = float3(bnoise(p+e.xyy)-n0, bnoise(p+e.yxy)-n0, bnoise(p+e.yyx)-n0)/e.x;
    n = normalize(n-d*2.5/sqrt(ds));
    return n;
}



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.1+ _XYZPos;                                             // ray origin

//    float2 p = fragCoord.xy/iResolution.xy-0.5;
//    float2 q = fragCoord.xy/iResolution.xy;
//    p.x*=iResolution.x/iResolution.y;
//    float2 mo = iMouse.xy / iResolution.xy-.5;
//    mo = (mo==float2(-.5))?mo=float2(-0.1,0.07):mo;
//    mo.x *= iResolution.x/iResolution.y;
    
//    float3 ro = float3(smoothstep(0.,1.,tri(time*.6)*2.)*0.1, smoothstep(0.,1.,tri(time*1.2)*2.)*0.05, -time*0.6);
//    ro.y -= height(ro.zx)+0.07;
//    mo.x += smoothstep(0.7,1.,sin(time*.35))-1.5 - smoothstep(-.7,-1.,sin(time*.35));
 
//    float3 eyedir = normalize(float3(cos(mo.x),mo.y*2.-0.2+sin(time*.75*1.37)*0.15,sin(mo.x)));
//    float3 rightdir = normalize(float3(cos(mo.x+1.5708),0.,sin(mo.x+1.5708)));
//    float3 updir = normalize(cross(rightdir,eyedir));
//    float3 rd=normalize((p.x*rightdir+p.y*updir)*1.+eyedir);
    
    ligt = normalize( float3(.5, .5, -.2) );
    
    float rz = march(ro,rd);
    
    float3 fogb = lerp(float3(.5, .4,.4), float3(1.,.9,.8), min(pow(max(dot(rd,ligt), 0.0), 1.5)*1.25, 1.0));
    float3 col = fogb;
    
    if ( rz < FAR )
    {
        float3 pos = ro+rz*rd;
        float3 nor= normal( pos );
        float d = distance(pos,ro);
        nor = bump(pos,nor,d);
        float shd = shadow(pos,ligt,0.1,3.);
        float dif = clamp( dot( nor, ligt ), 0.0, 1.0 )*shd;
        float spe = pow(clamp( dot( reflect(rd,nor), ligt ), 0.0, 1.0 ),3.)*shd;
        float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 1.5 );
        float3 brdf = dif*float3(1.00,1,1)+abs(nor.y)*.4;
        col = clamp(lerp(float3(.7,0.4,.3),float3(.3, 0.1, 0.1),(pos.y+.5)*.25), .0, 1.0);
        col *= (sin(bnoise(pos*.1)*250.)*0.5+0.5);
        col = col*brdf + spe*fre;// + fre*float3(.4,.4,0.4)*.5*crv;
    }
    
    //ordinary distance fog first
    col = lerp(col, fogb, smoothstep(FAR-10.,FAR,rz));
    
    //then volumetric fog
    col = fog(col, ro, rd, rz);
    
    //post...

    col = pow(col,float3(0.8,0.8,0.8));
    col = smoothstep(0.0, 1.0, col);
    
//    col *= .5+.5*pow(70. *q.x*q.y*(1.0-q.x)*(1.0-q.y), .2);
    
    
    fragColor = float4( col  * smoothstep(0.0, 3.0, time), 1.0 );

                return fragColor;
            }

            ENDCG
        }
    }
}

